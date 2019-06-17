// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "image_loader.h"
#include "CachedInterpolation.h"
#include <jpeglib.h>
#include "local_filesystem.h"
#include "jpeg_mem.h"
#include "Callback.h"

template <typename T>
void ResizeImageInMemory(const T* input_data, float* output_data, int in_height, int in_width, int out_height,
                         int out_width, int channels) {
  float height_scale = CalculateResizeScale(in_height, out_height, false);
  float width_scale = CalculateResizeScale(in_width, out_width, false);

  std::vector<CachedInterpolation> ys(out_height + 1);
  std::vector<CachedInterpolation> xs(out_width + 1);

  // Compute the cached interpolation weights on the x and y dimensions.
  compute_interpolation_weights(out_height, in_height, height_scale, ys.data());
  compute_interpolation_weights(out_width, in_width, width_scale, xs.data());

  // Scale x interpolation weights to avoid a multiplication during iteration.
  for (int i = 0; i < xs.size(); ++i) {
    xs[i].lower *= channels;
    xs[i].upper *= channels;
  }

  const int64_t in_row_size = in_width * channels;
  const int64_t in_batch_num_values = in_height * in_row_size;
  const int64_t out_row_size = out_width * channels;

  const T* input_b_ptr = input_data;
  float* output_y_ptr = output_data;
  const int batch_size = 1;

  if (channels == 3) {
    for (int b = 0; b < batch_size; ++b) {
      for (int64_t y = 0; y < out_height; ++y) {
        const T* ys_input_lower_ptr = input_b_ptr + ys[y].lower * in_row_size;
        const T* ys_input_upper_ptr = input_b_ptr + ys[y].upper * in_row_size;
        const float ys_lerp = ys[y].lerp;
        for (int64_t x = 0; x < out_width; ++x) {
          const int64_t xs_lower = xs[x].lower;
          const int64_t xs_upper = xs[x].upper;
          const float xs_lerp = xs[x].lerp;

          // Read channel 0.
          const float top_left0(ys_input_lower_ptr[xs_lower + 0]);
          const float top_right0(ys_input_lower_ptr[xs_upper + 0]);
          const float bottom_left0(ys_input_upper_ptr[xs_lower + 0]);
          const float bottom_right0(ys_input_upper_ptr[xs_upper + 0]);

          // Read channel 1.
          const float top_left1(ys_input_lower_ptr[xs_lower + 1]);
          const float top_right1(ys_input_lower_ptr[xs_upper + 1]);
          const float bottom_left1(ys_input_upper_ptr[xs_lower + 1]);
          const float bottom_right1(ys_input_upper_ptr[xs_upper + 1]);

          // Read channel 2.
          const float top_left2(ys_input_lower_ptr[xs_lower + 2]);
          const float top_right2(ys_input_lower_ptr[xs_upper + 2]);
          const float bottom_left2(ys_input_upper_ptr[xs_lower + 2]);
          const float bottom_right2(ys_input_upper_ptr[xs_upper + 2]);

          // Compute output.
          output_y_ptr[x * channels + 0] =
              compute_lerp(top_left0, top_right0, bottom_left0, bottom_right0, xs_lerp, ys_lerp);
          output_y_ptr[x * channels + 1] =
              compute_lerp(top_left1, top_right1, bottom_left1, bottom_right1, xs_lerp, ys_lerp);
          output_y_ptr[x * channels + 2] =
              compute_lerp(top_left2, top_right2, bottom_left2, bottom_right2, xs_lerp, ys_lerp);
        }
        output_y_ptr += out_row_size;
      }
      input_b_ptr += in_batch_num_values;
    }
  } else {
    for (int b = 0; b < batch_size; ++b) {
      for (int64_t y = 0; y < out_height; ++y) {
        const T* ys_input_lower_ptr = input_b_ptr + ys[y].lower * in_row_size;
        const T* ys_input_upper_ptr = input_b_ptr + ys[y].upper * in_row_size;
        const float ys_lerp = ys[y].lerp;
        for (int64_t x = 0; x < out_width; ++x) {
          auto xs_lower = xs[x].lower;
          auto xs_upper = xs[x].upper;
          auto xs_lerp = xs[x].lerp;
          for (int c = 0; c < channels; ++c) {
            const float top_left(ys_input_lower_ptr[xs_lower + c]);
            const float top_right(ys_input_lower_ptr[xs_upper + c]);
            const float bottom_left(ys_input_upper_ptr[xs_lower + c]);
            const float bottom_right(ys_input_upper_ptr[xs_upper + c]);
            output_y_ptr[x * channels + c] =
                compute_lerp(top_left, top_right, bottom_left, bottom_right, xs_lerp, ys_lerp);
          }
        }
        output_y_ptr += out_row_size;
      }
      input_b_ptr += in_batch_num_values;
    }
  }
}

template void ResizeImageInMemory(const float* input_data, float* output_data, int in_height, int in_width,
                                  int out_height, int out_width, int channels);

template void ResizeImageInMemory(const uint8_t* input_data, float* output_data, int in_height, int in_width,
                                  int out_height, int out_width, int channels);

// https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
// function: preprocess_for_eval
void InceptionPreprocessing::operator()(void* input_data, void* output_data)  {

  TCharString& file_name_ = *reinterpret_cast<TCharString*>(input_data);
  UncompressFlags flags;
  flags.components = channels_;
  // The TensorFlow-chosen default for jpeg decoding is IFAST, sacrificing
  // image quality for speed.
  flags.dct_method = JDCT_IFAST;
  size_t file_len;
  void* file_data;
  Callback c;
  ReadFileAsString(file_name_.c_str(), file_data, file_len, c);
  int width;
  int height;
  int channels;
  std::unique_ptr<uint8_t[]> decompressed_image(
      Uncompress(file_data, file_len, flags, &width, &height, &channels, nullptr));
  if (c.f) c.f(c.param);

  if (channels != channels_) {
    std::ostringstream oss;
    oss << "input format error, expect 3 channels, got " << channels;
    throw std::runtime_error(oss.str());
  }

  // cast uint8 to float
  // See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/image_ops_impl.py of
  // tf.image.convert_image_dtype
  std::vector<float> float_file_data(height * width * channels);
  {
    auto p = decompressed_image.get();
    for (size_t i = 0; i != float_file_data.size(); ++i) {
      float_file_data[i] = static_cast<float>(p[i]) / 255;
    }
  }

  // TODO: crop it
  auto output_data_ = reinterpret_cast<float*>(output_data);
  ResizeImageInMemory(float_file_data.data(), output_data_, height, width, out_height_, out_width_, channels);
  size_t output_data_len = channels_ * out_height_ * out_width_;
  for (size_t i = 0; i != output_data_len; ++i) {
    output_data_[i] = (output_data_[i] - 0.5) * 2;
  }
}
