//
// Created by Brian Rodriguez on 8/27/23.
//
#pragma once

#include <cstddef>

namespace luna::operators {
void im2col(float* input, size_t batch_size, size_t height, size_t width, size_t channels, size_t kernel_height, size_t kernel_width);
}
