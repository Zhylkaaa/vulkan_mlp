//
// Created by Dima Zhylko on 12/05/2020.
//

#ifndef VULKAN_PERCEPTRON_DENSE_H
#define VULKAN_PERCEPTRON_DENSE_H
#include "layer.h"
#include <string>
#include "vulkan_init.h"
#include <iostream>

class DenseLayer: public Layer{
    VkBuffer weight;
    VkBuffer bias;

    VkBuffer d_weight;
    VkBuffer d_bias;

    float scale;

    std::string initializer{};

    struct dims{
        uint32_t batch_size;
        uint32_t inp_dim;
        uint32_t output_dim;
    } dim;

public:
    DenseLayer(VkDevice device, uint32_t queueFamilyIndex, VkPhysicalDevice physicalDevice,
               int batch_size, int input_dim, int output_dim, VkBuffer input, float scale=10, const std::string& initializer="xavier");

    void forward_initialize(VkQueue& queue) override;
    void forward(VkQueue& queue) override;
    void backward(VkQueue& queue) override;

    VkBuffer& get_bias();
    VkBuffer& get_weight();

    [[nodiscard]] const dims& get_dims() const {return dim;};
    uint32_t get_output_dim() override {return dim.output_dim;}

    uint64_t get_output_offset() override {return offsets[2];}
};
#endif //VULKAN_PERCEPTRON_DENSE_H
