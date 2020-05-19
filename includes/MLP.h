//
// Created by Dima Zhylko on 12/05/2020.
//

#ifndef VULKAN_PERCEPTRON_MLP_H
#define VULKAN_PERCEPTRON_MLP_H
#include <vector>
#include <string>
#include "layer.h"
#include "dense.h"
#include "relu.h"
#include "softmax.h"
#include <vulkan_init.h>
#include <iostream>

class MLP {
    friend class Trainer;
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

    void eval_batch(std::vector<std::vector<float>>& batch, VkCommandBuffer& evalCommandBuffer,
                    VkDeviceMemory evalDeviceMemory, std::vector<uint64_t>& eval_offsets, uint32_t& correct_predictions,
                    std::vector<std::vector<float>>& true_labels);

public:
    MLP();

    MLP(uint32_t input_size, uint32_t batch_size, const std::vector<int>& layer_dims, const std::vector<std::string>& activations);

    void add(int layer_dim, const std::string& activation, uint32_t input_size=0, uint32_t batch_size=0);

    void forward_initialize();
    void forward(const std::vector<std::vector<float>>& batch);
    void backward_initialize(VkBuffer& d_out);
    void backward();

    VkBuffer& get_output() {return layers[layers.size()-1]->get_output();}
    uint32_t get_output_dim() {return layers[layers.size()-1]->get_output_dim();}

    VkDeviceMemory& get_output_memory() {return layers[layers.size()-1]->get_forward_device_memory();}

    VkBuffer& get_d_output() {return layers[layers.size()-1]->get_d_output();}

    uint64_t get_output_offset(){return layers[layers.size()-1]->get_output_offset();}

    uint32_t get_batch_size(){return batch_size;}
    uint32_t get_layer_count(){return layers.size();}

    std::vector<std::pair<Tensor, Tensor>> get_trainable_parameters();

    float evaluate(std::vector<std::vector<float>>& X, std::vector<std::vector<float>>& y);

    ~MLP();

    // DEBUG
    VkDevice &get_device() {return device;}
    VkPhysicalDevice& get_physicalDevice() {return physicalDevice;}
    uint32_t get_queue_index() {return queueFamilyIndex;}
};

#endif //VULKAN_PERCEPTRON_MLP_H
