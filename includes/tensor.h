//
// Created by @Zhylkaaa on 17/05/2020.
//

#ifndef VULKAN_PERCEPTRON_TENSOR_H
#define VULKAN_PERCEPTRON_TENSOR_H

#include <vulkan/vulkan.h>

class Tensor {
    VkBuffer buffer;

public:
    struct dims{
        uint32_t height;
        uint32_t width;
    };

    uint64_t get_elements_count() const {return dim.height * dim.width;}
    VkBuffer& get_buffer() {return buffer;}
    uint32_t get_dims_byte_size() const {return sizeof(dims);}
    dims get_dims() {return dim;}

    void set_height(uint32_t height) {dim.height = height;}
    void set_width(uint32_t width) {dim.width = width;}

    // TODO: refactor
    uint32_t get_width() const {return dim.width;}
    uint32_t get_height() const {return dim.height;}
private:
    dims dim;
};

#endif //VULKAN_PERCEPTRON_TENSOR_H
