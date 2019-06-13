// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
void CropImpl(
    const T* input_data,
    const int src_start_x,
    const int src_start_y,
    const int src_w,
    const int src_hw,
    const fast_divmod& fdm_dst_w,
    const fast_divmod& fdm_dst_hw,
    T* output_data,
    const size_t N);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime