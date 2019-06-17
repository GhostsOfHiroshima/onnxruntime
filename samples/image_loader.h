// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include <vector>
#include <string>
#include "CachedInterpolation.h"
#include "parallel_task_callback.h"
#include <onnxruntime/core/session/onnxruntime_c_api.h>

inline void compute_interpolation_weights(const int64_t out_size, const int64_t in_size, const float scale,
                                          CachedInterpolation* interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  for (int64_t i = out_size - 1; i >= 0; --i) {
    const float in = i * scale;
    interpolation[i].lower = static_cast<int64_t>(in);
    interpolation[i].upper = std::min(interpolation[i].lower + 1, in_size - 1);
    interpolation[i].lerp = in - interpolation[i].lower;
  }
}

/**
 * Computes the bilinear interpolation from the appropriate 4 float points
 * and the linear interpolation weights.
 */
inline float compute_lerp(const float top_left, const float top_right, const float bottom_left,
                          const float bottom_right, const float x_lerp, const float y_lerp) {
  const float top = top_left + (top_right - top_left) * x_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}

template <typename T>
void ResizeImageInMemory(const T* input_data, float* output_data, int in_height, int in_width, int out_height,
                         int out_width, int channels);

/**
 * CalculateResizeScale determines the float scaling factor.
 * @param in_size
 * @param out_size
 * @param align_corners If true, the centers of the 4 corner pixels of the input and output tensors are aligned,
 *                        preserving the values at the corner pixels
 * @return
 */
inline float CalculateResizeScale(int64_t in_size, int64_t out_size, bool align_corners) {
  return (align_corners && out_size > 1) ? (in_size - 1) / static_cast<float>(out_size - 1)
                                         : in_size / static_cast<float>(out_size);
}

class RunnableTask : public std::unary_function<void, void> {
 public:
  virtual void operator()() = 0;
  virtual ~RunnableTask() = default;
};

class DataProcessing {
 public:
  virtual void operator()(void* input_data, void* output_data) = 0;
};

class InceptionPreprocessing : public DataProcessing {
 private:
  int out_height_;
  int out_width_;
  int channels_;
 public:
  InceptionPreprocessing(int out_height, int out_width, int channels):
      out_height_(out_height),out_width_(out_width),
      channels_(channels){}

  void operator()(void* input_data, void* output_data) override;
};
