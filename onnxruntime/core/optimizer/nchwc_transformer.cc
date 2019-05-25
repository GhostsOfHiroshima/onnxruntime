// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/nchwc_transformer.h"
#include "core/mlas/inc/mlas.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

// Rewrite Conv/FusedConv as NchwcConv with additional nodes to reorder input and output.
class NchwcConvPoolTransformer : public onnxruntime::GraphTransformer {
 public:
  NchwcConvPoolTransformer() noexcept : onnxruntime::GraphTransformer("NchwcConvPoolTransformer") {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    std::deque<onnxruntime::NodeIndex> removed_nodes;

    GraphViewer graph_viewer(graph);
    for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
      auto& node = *graph.GetNode(index);
      ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level));

      if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Conv", {1}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(node, "FusedConv", {1}, kMSDomain)) {

        auto& conv_inputs = node.MutableInputDefs();
        auto& conv_outputs = node.MutableOutputDefs();

        // Require that the weights tensor be static.
        const ONNX_NAMESPACE::TensorProto* conv_W_tensor_proto = nullptr;
        if (!graph.GetInitializedTensor(conv_inputs[1]->Name(), conv_W_tensor_proto)) {
          continue;
        }
        if ((conv_W_tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) ||
            (conv_W_tensor_proto->dims_size() != 4)) {
          continue;
        }

        const int64_t output_channels = conv_W_tensor_proto->dims(0);
        const int64_t input_channels = conv_W_tensor_proto->dims(1);

        int64_t group_count;
        const onnxruntime::NodeAttributes& conv_attributes = node.GetAttributes();
        const ONNX_NAMESPACE::AttributeProto* group_attr = &(conv_attributes.find("group")->second);
        if (group_attr != nullptr &&
            group_attr->type() == AttributeProto_AttributeType_INT &&
            group_attr->has_i()) {
          group_count = group_attr->i();
        } else {
          group_count = 1;
        }

        const size_t nchwc_block_size = MlasNchwcGetBlockSize();

        bool do_reorder_input = true;
        bool reorder_filter_OIHWBo = false;

        if (group_count > 1) {
          if (input_channels == 1 && output_channels == group_count) {
            // Depthwise convolution.
            reorder_filter_OIHWBo = true;
          } else if ((input_channels % nchwc_block_size) != 0) {
            continue;
          } else {
            if (((output_channels % group_count) != 0) ||
                (((output_channels / group_count) % nchwc_block_size) != 0)) {
              continue;
            }
          }
        } else {
          if (static_cast<size_t>(input_channels) < nchwc_block_size) {
            // Use NCHW input buffer directly.
            reorder_filter_OIHWBo = true;
            do_reorder_input = false;
          } else if ((input_channels % nchwc_block_size) != 0) {
            continue;
          }
        }

        if ((output_channels % nchwc_block_size) != 0) {
          continue;
        }

        ONNX_NAMESPACE::TensorProto new_conv_W_tensor_proto(*conv_W_tensor_proto);

        auto conv_W = std::make_unique<Initializer>(conv_W_tensor_proto);
        std::vector<float> reordered_filter(conv_W->size());

        // Reorder the weights tensor statically.
        if (reorder_filter_OIHWBo) {
          MlasReorderFilterOIHWBo(conv_W->dims().data(), conv_W->data<float>(), reordered_filter.data());
        } else {
          MlasReorderFilterOIHWBiBo(conv_W->dims().data(), conv_W->data<float>(), reordered_filter.data());
        }

        new_conv_W_tensor_proto.set_raw_data(reordered_filter.data(), reordered_filter.size() * sizeof(float));

        graph.RemoveInitializedTensor(conv_inputs[1]->Name());
        graph.AddInitializedTensor(new_conv_W_tensor_proto);

        std::vector<Node::EdgeEnd> input_edges;
        for (auto it = node.InputEdgesBegin(); it != node.InputEdgesEnd(); ++it) {
          input_edges.push_back(*it);
        }
        for (auto& edge : input_edges) {
          graph.RemoveEdge(edge.GetNode().Index(), node.Index(), edge.GetSrcArgIndex(), edge.GetDstArgIndex());
        }

        std::vector<Node::EdgeEnd> output_edges;
        for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
          output_edges.push_back(*it);
        }
        for (auto& edge : output_edges) {
          graph.RemoveEdge(node.Index(), edge.GetNode().Index(), edge.GetSrcArgIndex(), edge.GetDstArgIndex());
        }

        // Reorder the input tensor.
        if (do_reorder_input) {
          auto input_original_arg = conv_inputs[0];
          std::string input_reorder_def_name = graph.GenerateNodeArgName("reorderInput");
          auto* input_reorder_arg = &graph.GetOrCreateNodeArg(input_reorder_def_name, nullptr);
          Node& reorder_input_node = graph.AddNode(graph.GenerateNodeName("ReorderInput"),
                                                   "ReorderInput",
                                                   "ReorderInput",
                                                   std::vector<NodeArg*>{input_original_arg},
                                                   std::vector<NodeArg*>{input_reorder_arg},
                                                   nullptr,
                                                   kMSDomain);
          reorder_input_node.SetExecutionProviderType(node.GetExecutionProviderType());
          conv_inputs[0] = input_reorder_arg;
        }

        // Reorder the output tensor.
        auto output_original_arg = conv_outputs[0];
        std::string output_reorder_def_name = graph.GenerateNodeArgName("reorderOutput");
        auto* output_reorder_arg = &graph.GetOrCreateNodeArg(output_reorder_def_name, nullptr);
        Node& reorder_output_node = graph.AddNode(graph.GenerateNodeName("ReorderOutput"),
                                                  "ReorderOutput",
                                                  "ReorderOutput",
                                                  std::vector<NodeArg*>{output_reorder_arg},
                                                  std::vector<NodeArg*>{output_original_arg},
                                                  nullptr,
                                                  kMSDomain);
        reorder_output_node.SetExecutionProviderType(node.GetExecutionProviderType());
        conv_outputs[0] = output_reorder_arg;

        // Create the replacement NchwcConv node.
        std::string nchwc_conv_name = graph.GenerateNodeName("NchwcConv");
        Node& nchwc_conv_node = graph.AddNode(output_original_arg->Name() + "_nchwc",
                                              "NchwcConv",
                                              nchwc_conv_name,
                                              conv_inputs,
                                              conv_outputs,
                                              &node.GetAttributes(),
                                              kMSDomain);
        nchwc_conv_node.SetExecutionProviderType(node.GetExecutionProviderType());

        removed_nodes.push_front(node.Index());
        continue;
      }

      if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "MaxPool", {1, 8, 10})) {

        auto& pool_inputs = node.MutableInputDefs();
        auto& pool_outputs = node.MutableOutputDefs();

        // Don't support the index tensor output.
        if (pool_outputs.size() > 1) {
          continue;
        }

        std::vector<Node::EdgeEnd> input_edges;
        for (auto it = node.InputEdgesBegin(); it != node.InputEdgesEnd(); ++it) {
          input_edges.push_back(*it);
        }
        for (auto& edge : input_edges) {
          graph.RemoveEdge(edge.GetNode().Index(), node.Index(), edge.GetSrcArgIndex(), edge.GetDstArgIndex());
        }

        std::vector<Node::EdgeEnd> output_edges;
        for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
          output_edges.push_back(*it);
        }
        for (auto& edge : output_edges) {
          graph.RemoveEdge(node.Index(), edge.GetNode().Index(), edge.GetSrcArgIndex(), edge.GetDstArgIndex());
        }

        // Reorder the input tensor.
        auto input_original_arg = pool_inputs[0];
        std::string input_reorder_def_name = graph.GenerateNodeArgName("reorderInput");
        auto* input_reorder_arg = &graph.GetOrCreateNodeArg(input_reorder_def_name, nullptr);
        Node& reorder_input_node = graph.AddNode(graph.GenerateNodeName("ReorderInput"),
                                                 "ReorderInput",
                                                 "ReorderInput",
                                                 std::vector<NodeArg*>{input_original_arg},
                                                 std::vector<NodeArg*>{input_reorder_arg},
                                                 nullptr,
                                                 kMSDomain);
        reorder_input_node.SetExecutionProviderType(node.GetExecutionProviderType());
        pool_inputs[0] = input_reorder_arg;

        // Reorder the output tensor.
        auto output_original_arg = pool_outputs[0];
        std::string output_reorder_def_name = graph.GenerateNodeArgName("reorderOutput");
        auto* output_reorder_arg = &graph.GetOrCreateNodeArg(output_reorder_def_name, nullptr);
        Node& reorder_output_node = graph.AddNode(graph.GenerateNodeName("ReorderOutput"),
                                                  "ReorderOutput",
                                                  "ReorderOutput",
                                                  std::vector<NodeArg*>{output_reorder_arg},
                                                  std::vector<NodeArg*>{output_original_arg},
                                                  nullptr,
                                                  kMSDomain);
        reorder_output_node.SetExecutionProviderType(node.GetExecutionProviderType());
        pool_outputs[0] = output_reorder_arg;

        // Create the replacement NchwcConv node.
        std::string nchwc_pool_name = graph.GenerateNodeName("NchwcMaxPool");
        Node& nchwc_pool_node = graph.AddNode(output_original_arg->Name() + "_nchwc",
                                              "NchwcMaxPool",
                                              nchwc_pool_name,
                                              pool_inputs,
                                              pool_outputs,
                                              &node.GetAttributes(),
                                              kMSDomain);
        nchwc_pool_node.SetExecutionProviderType(node.GetExecutionProviderType());

        removed_nodes.push_front(node.Index());
        continue;
      }
    }

    for (auto node : removed_nodes) {
      graph.RemoveNode(node);
    }

    if (!removed_nodes.empty()) {
      modified = true;
    }

    return Status::OK();
  }
};

// Rewrites sequences of ReorderOutput->Node to Node->ReorderOutput in order to
// encourage later fusions and reordering cancelations.
class NchwcMoveReorderOutputsLater : public onnxruntime::GraphTransformer {
 public:
  NchwcMoveReorderOutputsLater() noexcept : onnxruntime::GraphTransformer("NchwcMoveReorderOutputsLater") {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    std::deque<onnxruntime::NodeIndex> removed_nodes;

    GraphViewer graph_viewer(graph);
    for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
      auto& node = *graph.GetNode(index);
      ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level));

      if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Sum", {8}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(node, "Relu", {6}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(node, "Clip", {6}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(node, "Concat", {4})) {

        // BUGBUG: Concat should only do this if the input blocks are fully aligned...

        auto& node_inputs = node.MutableInputDefs();
        auto& node_outputs = node.MutableOutputDefs();

        // Test if all of the inputs are ReorderOutput nodes.
        bool all_inputs_reorder_output = true;
        for (auto it = node.InputNodesBegin(); it != node.InputNodesEnd(); ++it) {
          auto& input_node = *it;
          if (!graph_utils::IsSupportedOptypeVersionAndDomain(input_node, "ReorderOutput", {1}, kMSDomain) ||
              (input_node.GetInputEdgesCount() != 1) || (input_node.GetOutputEdgesCount() != 1)) {
            all_inputs_reorder_output = false;
            break;
          }
        }
        if (!all_inputs_reorder_output) continue;

        // Capture the array of input ReorderOutput edges.
        std::vector<Node::EdgeEnd> reorder_edges;
        for (auto it = node.InputEdgesBegin(); it != node.InputEdgesEnd(); ++it) {
          reorder_edges.push_back(*it);
        }

        // Remove the ReorderOutput edges from the graph.
        for (auto& reorder_edge : reorder_edges) {
          auto& reorder_output_node = *graph.GetNode(reorder_edge.GetNode().Index());
          auto nchwc_edge = *reorder_output_node.InputEdgesBegin();
          graph.RemoveEdge(nchwc_edge.GetNode().Index(), reorder_output_node.Index(), nchwc_edge.GetSrcArgIndex(), nchwc_edge.GetDstArgIndex());
          graph.RemoveEdge(reorder_output_node.Index(), node.Index(), reorder_edge.GetSrcArgIndex(), reorder_edge.GetDstArgIndex());
          node_inputs[reorder_edge.GetDstArgIndex()] = reorder_output_node.MutableInputDefs()[nchwc_edge.GetDstArgIndex()];
          graph.AddEdge(nchwc_edge.GetNode().Index(), node.Index(), nchwc_edge.GetSrcArgIndex(), reorder_edge.GetDstArgIndex());
          removed_nodes.push_front(reorder_output_node.Index());
        }

        auto output_original_arg = node_outputs[0];
        std::string output_reorder_def_name = graph.GenerateNodeArgName("reorderOutput");
        auto* output_reorder_arg = &graph.GetOrCreateNodeArg(output_reorder_def_name, output_original_arg->TypeAsProto());
        Node& new_node = graph.AddNode(graph.GenerateNodeName("ReorderOutput"),
                                       "ReorderOutput",
                                       "ReorderOutput",
                                       std::vector<NodeArg*>{output_reorder_arg},
                                       std::vector<NodeArg*>{output_original_arg},
                                       nullptr,
                                       kMSDomain);
        new_node.SetExecutionProviderType(node.GetExecutionProviderType());
        node_outputs[0] = output_reorder_arg;
      }
    }

    for (auto node : removed_nodes) {
      graph.RemoveNode(node);
    }

    if (!removed_nodes.empty()) {
      modified = true;
    }

    return Status::OK();
  }
};

// Removes unneeded ReorderOutput->ReorderInput nodes.
class NchwcReorderElimination : public onnxruntime::GraphTransformer {
 public:
  NchwcReorderElimination() noexcept : onnxruntime::GraphTransformer("NchwcReorderElimination") {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    std::deque<onnxruntime::NodeIndex> removed_nodes;

    GraphViewer graph_viewer(graph);
    for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
      auto& node = *graph.GetNode(index);
      ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level));

      if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "ReorderOutput", {1}, kMSDomain)) {

        // Capture the array of output ReorderOutput edges.
        std::vector<Node::EdgeEnd> reorder_edges;
        for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
          reorder_edges.push_back(*it);
        }

        auto input_edge = *node.InputEdgesBegin();
        auto& input_node = *graph.GetNode(input_edge.GetNode().Index());

        for (auto& reorder_edge : reorder_edges) {

          const auto& next_node = reorder_edge.GetNode();
          if ((next_node.GetOutputEdgesCount() != 1) ||
              !graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "ReorderInput", {1}, kMSDomain)) {
            continue;
          }

          auto output_edge = *next_node.OutputEdgesBegin();
          auto& output_node = *graph.GetNode(output_edge.GetNode().Index());

          graph.RemoveEdge(node.Index(), next_node.Index(), reorder_edge.GetSrcArgIndex(), reorder_edge.GetDstArgIndex());
          graph.RemoveEdge(next_node.Index(), output_node.Index(), output_edge.GetSrcArgIndex(), output_edge.GetDstArgIndex());

          output_node.MutableInputDefs()[output_edge.GetDstArgIndex()] = input_node.MutableOutputDefs()[input_edge.GetSrcArgIndex()];

          removed_nodes.push_front(next_node.Index());
        }

        if (node.GetOutputEdgesCount() == 0) {
          removed_nodes.push_front(node.Index());
        }
      }
    }

    for (auto node : removed_nodes) {
      graph.RemoveNode(node);
    }

    if (!removed_nodes.empty()) {
      modified = true;
    }

    return Status::OK();
  }
};

// Rewrites NchwcConv->Sum such that the second input of the Sum is fed into NchwcConv,
// which can accumulate into the buffer as an inplace operation.
class NchwcConvSumFusion : public onnxruntime::GraphTransformer {
 public:
  NchwcConvSumFusion() noexcept : onnxruntime::GraphTransformer("NchwcConvSumFusion") {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    std::deque<onnxruntime::NodeIndex> removed_nodes;

    GraphViewer graph_viewer(graph);
    for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
      auto& node = *graph.GetNode(index);
      ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level));

      if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Sum", {8}) &&
          (node.GetInputEdgesCount() == 2)) {

        auto input_nodes = node.InputNodesBegin();
        auto& first_input_node = *input_nodes;
        ++input_nodes;
        auto& second_input_node = *input_nodes;

        auto can_fuse_input_node = [](const Node& input_node) {
          if ((input_node.GetOutputEdgesCount() == 1) &&
              graph_utils::IsSupportedOptypeVersionAndDomain(input_node, "NchwcConv", {1}, kMSDomain)) {
            const auto& attrs = input_node.GetAttributes();
            if (attrs.find("activation") == attrs.end()) {
              return true;
            }
            // BUGBUG: also verify that Sum not already fused...
          }
          return false;
        };

        NodeIndex conv_input_node_index;
        if (can_fuse_input_node(first_input_node)) {
          conv_input_node_index = first_input_node.Index();
        } else if (can_fuse_input_node(second_input_node)) {
          conv_input_node_index = second_input_node.Index();
        } else {
          continue;
        }

        auto& conv_input_node = *graph.GetNode(conv_input_node_index);

        std::vector<Node::EdgeEnd> input_edges;
        for (auto it = node.InputEdgesBegin(); it != node.InputEdgesEnd(); ++it) {
          input_edges.push_back(*it);
        }
        for (auto& edge : input_edges) {
          graph.RemoveEdge(edge.GetNode().Index(), node.Index(), edge.GetSrcArgIndex(), edge.GetDstArgIndex());
          if (&edge.GetNode() == &conv_input_node) {
            conv_input_node.MutableOutputDefs()[0] = node.MutableOutputDefs()[0];
          } else {
            if (conv_input_node.MutableInputDefs().size() < 4) {
              conv_input_node.MutableInputDefs().resize(4);
            }
            if (conv_input_node.MutableInputArgsCount().size() < 4) {
              conv_input_node.MutableInputArgsCount().resize(4);
            }
            auto& other_input_node = *graph.GetNode(edge.GetNode().Index());
            conv_input_node.MutableInputDefs()[3] = other_input_node.MutableOutputDefs()[edge.GetSrcArgIndex()];
            conv_input_node.MutableInputArgsCount()[3] = 1;
          }
        }

        std::vector<Node::EdgeEnd> output_edges;
        for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
          output_edges.push_back(*it);
        }
        for (auto& edge : output_edges) {
          graph.RemoveEdge(node.Index(), edge.GetNode().Index(), edge.GetSrcArgIndex(), edge.GetDstArgIndex());
        }

        removed_nodes.push_front(node.Index());
      }
    }

    for (auto node : removed_nodes) {
      graph.RemoveNode(node);
    }

    if (!removed_nodes.empty()) {
      modified = true;
    }

    return Status::OK();
  }
};

// Fuses NchwcConv->Relu nodes. These can occur if NchwcConvSumFusion was able to
// fuse.
class NchwcConvReluFusion : public onnxruntime::GraphTransformer {
 public:
  NchwcConvReluFusion() noexcept : onnxruntime::GraphTransformer("NchwcConvReluFusion") {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    std::deque<onnxruntime::NodeIndex> removed_nodes;

    GraphViewer graph_viewer(graph);
    for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
      auto& node = *graph.GetNode(index);
      ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level));

      if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Relu", {6})) {

        auto input_edge = *node.InputEdgesBegin();
        auto& input_node = *graph.GetNode(input_edge.GetNode().Index());
        const auto& attrs = input_node.GetAttributes();

        if (graph_utils::IsSupportedOptypeVersionAndDomain(input_node, "NchwcConv", {1}, kMSDomain) &&
            (input_node.GetOutputEdgesCount() == 1) &&
            (attrs.find("activation") == attrs.end())) {

          graph.RemoveEdge(input_edge.GetNode().Index(), node.Index(), input_edge.GetSrcArgIndex(), input_edge.GetDstArgIndex());

          std::vector<Node::EdgeEnd> output_edges;
          for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
            output_edges.push_back(*it);
          }
          for (auto& edge : output_edges) {
            graph.RemoveEdge(node.Index(), edge.GetNode().Index(), edge.GetSrcArgIndex(), edge.GetDstArgIndex());
          }

          input_node.MutableOutputDefs()[0] = node.MutableOutputDefs()[0];
          input_node.AddAttribute("activation", node.OpType());

          removed_nodes.push_front(node.Index());
        }
      }
    }

    for (auto node : removed_nodes) {
      graph.RemoveNode(node);
    }

    if (!removed_nodes.empty()) {
      modified = true;
    }

    return Status::OK();
  }
};


NchwcTransformer::NchwcTransformer() noexcept :
  onnxruntime::GraphTransformer("NchwcTransformer"), graph_transformer_mgr_{50} {

  // As implemented, these transforms can require a large number of steps to
  // reach a fully optimized graph (in particular, NchwcMoveReorderOutputsLater).
  graph_transformer_mgr_.Register(std::move(std::make_unique<NchwcConvPoolTransformer>()), TransformerLevel::Default);
  graph_transformer_mgr_.Register(std::move(std::make_unique<NchwcMoveReorderOutputsLater>()), TransformerLevel::Default);
  graph_transformer_mgr_.Register(std::move(std::make_unique<NchwcReorderElimination>()), TransformerLevel::Default);
  graph_transformer_mgr_.Register(std::move(std::make_unique<NchwcConvSumFusion>()), TransformerLevel::Default);
  graph_transformer_mgr_.Register(std::move(std::make_unique<NchwcConvReluFusion>()), TransformerLevel::Default);
}

Status NchwcTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  ORT_UNUSED_PARAMETER(modified);
  ORT_UNUSED_PARAMETER(graph_level);
  return graph_transformer_mgr_.ApplyTransformers(graph, TransformerLevel::Default);
}

}  // namespace onnxruntime
