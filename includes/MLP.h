//
// Created by Dima Zhylko on 12/05/2020.
//

#ifndef VULKAN_PERCEPTRON_MLP_H
#define VULKAN_PERCEPTRON_MLP_H
#include <vector>
#include <string>
#include "layer.h"
#include "dense.h"
#include "vulkan_init.h"
#include <iostream>

class MLP {
    std::vector<Layer*> layers{};
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice;
    uint32_t queueFamilyIndex;
    VkDevice device;
    VkQueue queue;
    VkDeviceMemory deviceMemory;

    VkBuffer input;
    VkBuffer d_input;
    std::vector<uint64_t> offsets;

    uint32_t batch_size;
    uint32_t input_size;
public:
    MLP();

    MLP(uint32_t input_size, uint32_t batch_size, const std::vector<int>& layer_dims, const std::vector<std::string>& activations);

    void add(int layer_dim, const std::string& activation, uint32_t input_size=0, uint32_t batch_size=0);

    void forward_initialize();
    void forward(const std::vector<std::vector<float>>& batch);
    void backward();

    VkBuffer& get_output() {return layers[layers.size()-1]->get_output();}

    VkDeviceMemory& get_memory() {return layers[layers.size()-1]->get_device_memory();}

    uint64_t get_output_offset(){return offsets[2];}
};

#endif //VULKAN_PERCEPTRON_MLP_H
