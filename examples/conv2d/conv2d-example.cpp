//
// Created by Brian Rodriguez on 6/17/23.
//
#include <luna/functional/conv2d.h>
#include <algorithm>
#include <memory>

void initialize_matrix(float* matrix, size_t size, float value, int skip = 1) {
  for (size_t i = 0; i < size; i += skip) {
    matrix[i] = value;
  }
}

int main() {
    size_t input_height = 28;
    size_t input_width = 28;
    size_t input_channels = 1;
    size_t kernel_height = 3;
    size_t kernel_width = 3;
    size_t output_channels = 10;
    size_t stride_height = 1;
    size_t stride_width = 1;
    size_t padding_height = 0;
    size_t padding_width = 0;
    size_t dilation_height = 1;
    size_t dilation_width = 1;
    size_t batch_size = 1;
    size_t groups = 1;

    size_t height_out = luna::functional::get_output_dim(input_height, kernel_height, stride_height, padding_height, dilation_height);
    size_t width_out = luna::functional::get_output_dim(input_width, kernel_width, stride_width, padding_width, dilation_width);
    auto* input = new float[batch_size * input_height * input_width * input_channels];
    auto* kernel = new float[kernel_height * kernel_width * input_channels * output_channels];
    auto* output = new float[batch_size * height_out * width_out * output_channels];
    initialize_matrix(input, batch_size * input_height * input_width * input_channels, 1, 2);
    std::fill_n(kernel, kernel_height * kernel_width * input_channels * output_channels, 10);
    luna::functional::conv_2d_simple(
        input,
        kernel,
        output,
        batch_size,
        input_height,
        input_width,
        input_channels,
        kernel_height,
        kernel_width,
        height_out,
        width_out,
        output_channels,
        stride_height,
        stride_width);


    return 0;
}