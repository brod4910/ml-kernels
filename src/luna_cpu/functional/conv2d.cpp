//
// Created by Brian Rodriguez on 7/27/23.
//
#include "luna_cpu/functional/conv2d.h"

namespace luna::functional {
void conv_2d(const float *input,
             const float *kernel,
             float *output,
             int batch_size,
             int input_height,
             int input_width,
             int input_channels,
             int kernel_height,
             int kernel_width,
             int output_height,
             int output_width,
             int output_channels,
             int stride_height,
             int stride_width,
             int padding_height,
             int padding_width,
             int dilation_height,
             int dilation_width,
             int groups) {
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t g = 0; g < groups; ++g) {
      for (size_t c = 0; c < output_channels; ++c) {
        for (size_t h = 0; h < output_height; ++h) {
          for (size_t w = 0; w < output_width; ++w) {
            float sum = 0.0f;
            for (size_t kc = 0; kc < input_channels; ++kc) {
              for (size_t kh = 0; kh < kernel_height; ++kh) {
                for (size_t kw = 0; kw < kernel_width; ++kw) {
                  size_t input_index =
                      get_index(b, kc, h * stride_height + kh * dilation_height,
                                w * stride_width + kw * dilation_width,
                                input_channels, input_height, input_width);
                  size_t kernel_index = get_index(0, c, kh, kw, input_channels,
                                                  kernel_height, kernel_width);
                  sum += input[input_index] * kernel[kernel_index];
                }
              }
            }
            size_t output_index = get_index(b, c, h, w, output_channels,
                                            output_height, output_width);
            output[output_index] = sum;
          }
        }
      }
    }
  }
}

void conv_2d_simple(const float *input, const float *kernel, float *output,
                    int batch_size, int input_height, int input_width,
                    int input_channels, int kernel_height, int kernel_width,
                    int output_height, int output_width, int output_channels,
                    int stride_height, int stride_width) {
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < output_channels; ++c) {
      for (size_t h = 0; h < output_height; ++h) {
        for (size_t w = 0; w < output_width; ++w) {
          float sum = 0.0f;
          for (size_t kc = 0; kc < input_channels; ++kc) {
            for (size_t kh = 0; kh < kernel_height; ++kh) {
              for (size_t kw = 0; kw < kernel_width; ++kw) {
                size_t input_index = get_index(b, kc, h * stride_height + kh, w * stride_width + kw, input_channels, input_height, input_width);
                size_t kernel_index = get_index(0, c, kh, kw, input_channels, kernel_height, kernel_width);
                sum += input[input_index] * kernel[kernel_index];
              }
            }
          }
          size_t output_index = get_index(b, c, h, w, output_channels, output_height, output_width);
          output[output_index] = sum;
        }
      }
    }
  }
}
}