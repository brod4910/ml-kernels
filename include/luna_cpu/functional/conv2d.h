//
// Created by Brian Rodriguez on 11/28/22.
//

#ifndef LUNA_CONV2D_H
#define LUNA_CONV2D_H

#include <cctype>
#include <stdexcept>

namespace luna::functional {
/*
 * Calculates the output dimension of a 2D convolutional layer
 *
 * @param input_size The size of the input
 * @param kernel_size The size of the kernel
 * @param stride The stride of the convolution
 * @param padding The padding of the convolution
 * @param dilation The dilation of the convolution
 *
 * @return The output dimension of the convolution
 */
size_t get_output_dim(size_t input_size, size_t kernel_size, size_t stride, size_t padding, size_t dilation) {
  return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
}

/*
 * Calculates the index of a 4D conv2d
 * @param b The batch dimension
 * @param h The height dimension
 * @param w The width dimension
 * @param c The channel dimension
 * @param height The height of the conv2d
 * @param width The width of the conv2d
 * @param channels The number of channels in the conv2d
 *
 * @return The index of the conv2d
 */
size_t get_index(size_t b, size_t c, size_t h, size_t w, size_t channels, size_t height, size_t width) {
  return b * channels * height * width + c * height * width + h * width + w;
}

/*
 * Calculates a simple 2D convolution of an input conv2d with a kernel. The convolution doesn't
 * support dilation, groups or padding.
 *
 * @param input The input conv2d
 * @param kernel The kernel
 * @param output The output conv2d
 * @param batch_size The size of the batch
 * @param input_height The height of the input conv2d
 * @param input_width The width of the input conv2d
 * @param input_channels The number of input channels
 * @param kernel_height The height of the kernel
 * @param kernel_width The width of the kernel
 * @param output_height The height of the output conv2d
 * @param output_width The width of the output conv2d
 * @param output_channels The number of output channels
 * @param stride_height The height of the stride
 * @param stride_width The width of the stride
 *
 * @return void
 */
void conv_2d_simple(
    const float *input,
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
    int stride_width);

/*
 * Calculates the 2D convolution of an input conv2d with a kernel
 *
 * @param input The input conv2d
 * @param kernel The kernel
 * @param output The output conv2d
 * @param input_height The height of the input conv2d
 * @param input_width The width of the input conv2d
 * @param kernel_height The height of the kernel
 * @param kernel_width The width of the kernel
 * @param output_height The height of the output conv2d
 * @param output_width The width of the output conv2d
 * @param stride_height The height of the stride
 * @param stride_width The width of the stride
 * @param padding_height The height of the padding
 * @param padding_width The width of the padding
 * @param dilation_height The height of the dilation
 * @param dilation_width The width of the dilation
 * @param groups The number of groups
 * @param batch_size The size of the batch
 * @param input_channels The number of input channels
 * @param output_channels The number of output channels
 *
 * @return void
 */
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
             int groups);
}// namespace luna::functional

#endif// LUNA_CONV2D_H
