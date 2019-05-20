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

class NchwcConvPoolTransformer : public GraphTransformer {
 public:
  NchwcConvPoolTransformer() noexcept : GraphTransformer("NchwcConvPoolTransformer") {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {
    std::deque<NodeIndex> removed_nodes;
    Node::EdgeSet reordered_output_edges;

    GraphViewer graph_viewer(graph);

    for (NodeIndex index : graph_viewer.GetNodesInTopologicalOrder()) {
      Node& node = *graph.GetNode(index);
      ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level));

      if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Conv", {1}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(node, "FusedConv", {1}, kMSDomain)) {
        auto& conv_inputs = node.MutableInputDefs();
        auto& conv_outputs = node.MutableOutputDefs();

        bool do_nchwc_conv = true;
        bool do_reorder_input = true;
        bool do_reorder_format1 = true;

        // Require that the weights tensor be static.
        const ONNX_NAMESPACE::TensorProto* conv_W_tensor_proto = nullptr;
        if (!graph.GetInitializedTensor(conv_inputs[1]->Name(), conv_W_tensor_proto)) {
          do_nchwc_conv = false;
        } else if ((conv_W_tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) ||
                   (conv_W_tensor_proto->dims_size() != 4)) {
          do_nchwc_conv = false;
        } else {
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

          if (group_count > 1) {
            if (input_channels == 1 && output_channels == group_count) {
              // Depthwise convolution needs alternate filter formatting.
              do_reorder_format1 = false;
            } else {
              if ((output_channels % group_count) != 0) {
                do_nchwc_conv = false;
              }
              if ((input_channels % 8) != 0 || ((output_channels / group_count) % 8) != 0) {
                do_nchwc_conv = false;
              }
            }
          } else {
            if (input_channels < 8) {
              // Use NCHW input buffer directly.
              do_reorder_input = false;
              do_reorder_format1 = false;
            } else if ((input_channels % 8) != 0) {
              do_nchwc_conv = false;
            }
          }

          if ((output_channels % 8) != 0) {
            do_nchwc_conv = false;
          }
        }
        // Reorder the weights tensor statically.
        if (do_nchwc_conv) {
          ONNX_NAMESPACE::TensorProto new_conv_W_tensor_proto(*conv_W_tensor_proto);

          auto conv_W = std::make_unique<Initializer>(conv_W_tensor_proto);
          std::vector<float> reordered_filter(conv_W->size());

          if (do_reorder_format1) {
            MlasConvReorderFilter(conv_W->dims().data(), conv_W->data<float>(), reordered_filter.data());
          } else {
            MlasConvReorderFilter2(conv_W->dims().data(), conv_W->data<float>(), reordered_filter.data());
          }

          new_conv_W_tensor_proto.set_raw_data(reordered_filter.data(), reordered_filter.size() * sizeof(float));

          graph.RemoveInitializedTensor(conv_inputs[1]->Name());
          graph.AddInitializedTensor(new_conv_W_tensor_proto);
        }

        bool reordered_inputs[3] = {false, false, false};  // recording if the inputs coming from a reordered output: X, W, B
        for (auto it = node.InputEdgesBegin(); it != node.InputEdgesEnd(); ++it) {
          Node::EdgeEnd corresponding_output_edge(node, (*it).GetSrcArgIndex(), (*it).GetDstArgIndex());
          auto reordered_input_it = reordered_output_edges.find(corresponding_output_edge);
          if (reordered_input_it != reordered_output_edges.end()) {
            reordered_inputs[(*reordered_input_it).GetDstArgIndex()] = true;
            reordered_output_edges.erase(corresponding_output_edge);
          }
        }

        // process conv inputs.
        if (do_nchwc_conv) {
          if (do_reorder_input) {
            std::vector<Node::EdgeEnd> input_edges;
            for (auto it = node.InputEdgesBegin(); it != node.InputEdgesEnd(); ++it) {
              input_edges.push_back(*it);
            }
            for (auto& edge : input_edges) {
              graph.RemoveEdge(edge.GetNode().Index(), node.Index(), edge.GetSrcArgIndex(), edge.GetDstArgIndex());
            }

            // Insert a ReorderInput node if input X is not from a reordered output,
            // otherwise both ReorderInput and ReorderOutput nodes are eliminated, no need to insert reorder nodes.
            if (!reordered_inputs[0]) {
              auto input_original_arg = conv_inputs[0];
              std::string input_reorder_def_name = graph.GenerateNodeArgName("reorderInput");
              auto* input_reorder_arg = &graph.GetOrCreateNodeArg(input_reorder_def_name, input_original_arg->TypeAsProto());
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
          }
        } else {  //Don't do nchwc conv, so insert a ReorderOutput node for input X that comes from a reordered output
          if (reordered_inputs[0]) {
            NodeArg* input_original_arg = conv_inputs[0];
            std::string input_reorder_def_name = graph.GenerateNodeArgName("reorderOutput");
            auto* input_reorder_arg = &graph.GetOrCreateNodeArg(input_reorder_def_name, input_original_arg->TypeAsProto());
            Node& reorder_input_node = graph.AddNode(graph.GenerateNodeName("ReorderOutput"),
                                                     "ReorderOutput",
                                                     "ReorderOutput",
                                                     std::vector<NodeArg*>{input_original_arg},
                                                     std::vector<NodeArg*>{input_reorder_arg},
                                                     nullptr,
                                                     kMSDomain);
            reorder_input_node.SetExecutionProviderType(node.GetExecutionProviderType());
            conv_inputs[0] = input_reorder_arg;
          }
        }

        //process conv outputs and replace conv with nchwc conv
        if (do_nchwc_conv) {
          //marked as reordered inputs
          for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
            reordered_output_edges.insert(*it);
          }

          std::string nchwc_conv_name = graph.GenerateNodeName("NchwcConv");
          Node& nchwc_conv_node = graph.AddNode(conv_outputs[0]->Name() + "_nchwc",
                                                "NchwcConv",
                                                nchwc_conv_name,
                                                conv_inputs,
                                                conv_outputs,
                                                &node.GetAttributes(),
                                                kMSDomain);
          nchwc_conv_node.SetExecutionProviderType(node.GetExecutionProviderType());

          //this op output is the graph output, insert a ReorderOutput Node
          if (node.GetOutputEdgesCount() == 0) {
            NodeArg* output_original_arg = conv_outputs[0];
            std::string output_reorder_def_name = graph.GenerateNodeArgName("reorderOutput");
            auto* output_reorder_arg = &graph.GetOrCreateNodeArg(output_reorder_def_name, output_original_arg->TypeAsProto());
            Node& reorder_output_node = graph.AddNode(graph.GenerateNodeName("ReorderOutput"),
                                                      "ReorderOutput",
                                                      "ReorderOutput",
                                                      std::vector<NodeArg*>{output_original_arg},
                                                      std::vector<NodeArg*>{output_reorder_arg},
                                                      nullptr,
                                                      kMSDomain);
            reorder_output_node.SetExecutionProviderType(node.GetExecutionProviderType());
            conv_outputs[0] = output_reorder_arg;
          }

          removed_nodes.push_front(node.Index());
        }
        continue;

        //TODO: extend to other pooling ops
      } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "MaxPool", {1, 8, 10})) {
        auto& pool_inputs = node.MutableInputDefs();
        auto& pool_outputs = node.MutableOutputDefs();

        bool do_nchwc_pooling = true;
        // Don't support the index tensor output.
        if (pool_outputs.size() > 1) {
          do_nchwc_pooling = false;
        }
        if (pool_inputs[0]->Shape()->dim_size() != 4) {
          do_nchwc_pooling = false;
        }

        bool reordered_inputs = false;  // check if the inputs coming from a reordered output
        for (auto it = node.InputEdgesBegin(); it != node.InputEdgesEnd(); ++it) {
          Node::EdgeEnd corresponding_output_edge(node, (*it).GetSrcArgIndex(), (*it).GetDstArgIndex());
          auto reordered_input_it = reordered_output_edges.find(corresponding_output_edge);
          if (reordered_input_it != reordered_output_edges.end()) {
            reordered_inputs = true;
            reordered_output_edges.erase(corresponding_output_edge);
          }
        }
        if (do_nchwc_pooling) {
          std::vector<Node::EdgeEnd> input_edges;
          for (auto it = node.InputEdgesBegin(); it != node.InputEdgesEnd(); ++it) {
            input_edges.push_back(*it);
          }
          for (auto& edge : input_edges) {
            graph.RemoveEdge(edge.GetNode().Index(), node.Index(), edge.GetSrcArgIndex(), edge.GetDstArgIndex());
          }

          // Reorder the input tensor.
          if (!reordered_inputs) {  //input not from a reordered node, insert a ReorderInput node
            auto input_original_arg = pool_inputs[0];
            std::string input_reorder_def_name = graph.GenerateNodeArgName("reorderInput");
            auto* input_reorder_arg = &graph.GetOrCreateNodeArg(input_reorder_def_name, input_original_arg->TypeAsProto());
            Node& reorder_input_node = graph.AddNode(graph.GenerateNodeName("ReorderInput"),
                                                     "ReorderInput",
                                                     "ReorderInput",
                                                     std::vector<NodeArg*>{input_original_arg},
                                                     std::vector<NodeArg*>{input_reorder_arg},
                                                     nullptr,
                                                     kMSDomain);
            reorder_input_node.SetExecutionProviderType(node.GetExecutionProviderType());
            pool_inputs[0] = input_reorder_arg;
          }

          //added output edge to reordered output edges
          for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
            reordered_output_edges.insert(*it);
          }

          // Create the replacement Nchwc Pool node.
          std::string nchwc_pool_name = graph.GenerateNodeName("NchwcMaxPool");
          Node& nchwc_pool_node = graph.AddNode(pool_outputs[0]->Name() + "_nchwc",
                                                "NchwcMaxPool",
                                                nchwc_pool_name,
                                                pool_inputs,
                                                pool_outputs,
                                                &node.GetAttributes(),
                                                kMSDomain);
          nchwc_pool_node.SetExecutionProviderType(node.GetExecutionProviderType());

          //this op output is the graph output, insert a ReorderOutput Node
          if (node.GetOutputEdgesCount() == 0) {
            NodeArg* output_original_arg = pool_outputs[0];
            std::string output_reorder_def_name = graph.GenerateNodeArgName("reorderOutput");
            auto* output_reorder_arg = &graph.GetOrCreateNodeArg(output_reorder_def_name, output_original_arg->TypeAsProto());
            Node& reorder_output_node = graph.AddNode(graph.GenerateNodeName("ReorderOutput"),
                                                      "ReorderOutput",
                                                      "ReorderOutput",
                                                      std::vector<NodeArg*>{output_original_arg},
                                                      std::vector<NodeArg*>{output_reorder_arg},
                                                      nullptr,
                                                      kMSDomain);
            reorder_output_node.SetExecutionProviderType(node.GetExecutionProviderType());
          }

          removed_nodes.push_front(node.Index());
        } else if (reordered_inputs) {  //don't do nchwc pooling, insert ReorderOutput node
          NodeArg* input_original_arg = pool_inputs[0];
          std::string input_reorder_def_name = graph.GenerateNodeArgName("reorderOutput");
          auto* input_reorder_arg = &graph.GetOrCreateNodeArg(input_reorder_def_name, input_original_arg->TypeAsProto());
          Node& reorder_input_node = graph.AddNode(graph.GenerateNodeName("ReorderOutput"),
                                                   "ReorderOutput",
                                                   "ReorderOutput",
                                                   std::vector<NodeArg*>{input_original_arg},
                                                   std::vector<NodeArg*>{input_reorder_arg},
                                                   nullptr,
                                                   kMSDomain);
          reorder_input_node.SetExecutionProviderType(node.GetExecutionProviderType());
          pool_inputs[0] = input_reorder_arg;
        }
        continue;
      }
      //TODO: extend to other element_wise ops.
      else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Sum", {8}) ||
               graph_utils::IsSupportedOptypeVersionAndDomain(node, "Relu", {6}) ||
               graph_utils::IsSupportedOptypeVersionAndDomain(node, "Concat", {4})) {
        bool all_inputs_reorder_output = true;
        Node::EdgeSet to_reorder_inputs;
        for (auto it = node.InputEdgesBegin(); it != node.InputEdgesEnd(); ++it) {
          Node::EdgeEnd corresponding_output_edge(node, (*it).GetSrcArgIndex(), (*it).GetDstArgIndex());
          if (reordered_output_edges.find(corresponding_output_edge) == reordered_output_edges.end()) {
            all_inputs_reorder_output = false;
          } else {
            to_reorder_inputs.insert(corresponding_output_edge);
            reordered_output_edges.erase(corresponding_output_edge);
          }
        }

        if (all_inputs_reorder_output) {
          //move reorder node from inputs to outputs
          if (node.GetOutputEdgesCount() == 0) {  //this node output is graph output, insert an output node
            auto& outputs = node.MutableOutputDefs();
            NodeArg* output_original_arg = outputs[0];
            std::string output_reorder_def_name = graph.GenerateNodeArgName("reorderOutput");
            auto* output_reorder_arg = &graph.GetOrCreateNodeArg(output_reorder_def_name, output_original_arg->TypeAsProto());
            Node& reorder_output_node = graph.AddNode(graph.GenerateNodeName("ReorderOutput"),
                                                      "ReorderOutput",
                                                      "ReorderOutput",
                                                      std::vector<NodeArg*>{output_original_arg},
                                                      std::vector<NodeArg*>{output_reorder_arg},
                                                      nullptr,
                                                      kMSDomain);
            reorder_output_node.SetExecutionProviderType(node.GetExecutionProviderType());
          } else {
            for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
              reordered_output_edges.insert(*it);
            }
          }
        } else {
          // Insert ReorderOutput
          for (Node::EdgeEnd edge : to_reorder_inputs) {
            auto nodearg_index = edge.GetDstArgIndex();
            NodeArg* input_original_arg = node.MutableInputDefs()[nodearg_index];
            std::string input_reorder_def_name = graph.GenerateNodeArgName("ReorderOutput");
            auto* input_reorder_arg = &graph.GetOrCreateNodeArg(input_reorder_def_name, input_original_arg->TypeAsProto());
            Node& reorder_input_node = graph.AddNode(graph.GenerateNodeName("ReorderOutput"),
                                                     "ReorderOutput",
                                                     "ReorderOutput",
                                                     std::vector<NodeArg*>{input_original_arg},
                                                     std::vector<NodeArg*>{input_reorder_arg},
                                                     nullptr,
                                                     kMSDomain);
            reorder_input_node.SetExecutionProviderType(node.GetExecutionProviderType());
            node.MutableInputDefs()[nodearg_index] = input_reorder_arg;
          }
        }
      } else {  //For undelayable nodes, insert ReorderOutput
        for (auto it = node.InputEdgesBegin(); it != node.InputEdgesEnd(); ++it) {
          Node::EdgeEnd corresponding_output_edge(node, (*it).GetSrcArgIndex(), (*it).GetDstArgIndex());
          if (reordered_output_edges.find(corresponding_output_edge) != reordered_output_edges.end()) {
            auto nodearg_index = (*it).GetDstArgIndex();
            NodeArg* input_original_arg = node.MutableInputDefs()[nodearg_index];
            std::string input_reorder_def_name = graph.GenerateNodeArgName("reorderOutput");
            auto* input_reorder_arg = &graph.GetOrCreateNodeArg(input_reorder_def_name, input_original_arg->TypeAsProto());
            Node& reorder_input_node = graph.AddNode(graph.GenerateNodeName("ReorderOutput"),
                                                     "ReorderOutput",
                                                     "ReorderOutput",
                                                     std::vector<NodeArg*>{input_original_arg},
                                                     std::vector<NodeArg*>{input_reorder_arg},
                                                     nullptr,
                                                     kMSDomain);
            reorder_input_node.SetExecutionProviderType(node.GetExecutionProviderType());
            node.MutableInputDefs()[nodearg_index] = input_reorder_arg;
            reordered_output_edges.erase(corresponding_output_edge);
          }
        }
      }
    }
    for (auto index : removed_nodes) {
      graph.RemoveNode(index);
    }

    if (!removed_nodes.empty()) {
      modified = true;
    }

    return Status::OK();
  }
};  // namespace onnxruntime

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
        if (node.InputEdgesBegin() != node.InputEdgesEnd()) {
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
    }

    for (auto node : removed_nodes) {
      graph.RemoveNode(node);
    }

    if (!removed_nodes.empty()) {
      modified = true;
    }

    return Status::OK();
  }
};  // namespace onnxruntime

NchwcTransformer::NchwcTransformer() noexcept : onnxruntime::GraphTransformer("NchwcTransformer"), graph_transformer_mgr_{50} {
  // As implemented, these transforms can require a large number of steps to
  // reach a fully optimized graph (in particular, NchwcMoveReorderOutputsLater).
  graph_transformer_mgr_.Register(std::move(std::make_unique<NchwcConvPoolTransformer>()), TransformerLevel::Default);
  graph_transformer_mgr_.Register(std::move(std::make_unique<NchwcConvSumFusion>()), TransformerLevel::Default);
  graph_transformer_mgr_.Register(std::move(std::make_unique<NchwcConvReluFusion>()), TransformerLevel::Default);
}

Status NchwcTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  ORT_UNUSED_PARAMETER(modified);
  ORT_UNUSED_PARAMETER(graph_level);
  return graph_transformer_mgr_.ApplyTransformers(graph, TransformerLevel::Default);
}

}  // namespace onnxruntime