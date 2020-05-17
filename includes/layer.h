//
// Created by Dima Zhylko on 12/05/2020.
//

#ifndef VULKAN_PERCEPTRON_LAYER_H
#define VULKAN_PERCEPTRON_LAYER_H

#include <vulkan/vulkan.h>
#include <vector>

class Layer {
protected:
    VkBuffer input;
    VkBuffer output;
    std::vector<uint64_t> forward_offsets{};
    std::vector<uint64_t> backward_offsets{};
    VkBuffer d_input;
    VkBuffer d_output;

    VkDevice device;
    uint32_t queueFamilyIndex;
    VkPhysicalDevice physicalDevice;

    VkDeviceMemory forwardDeviceMemory;

    VkPipeline forwardPipeline;
    VkPipelineLayout forwardPipelineLayout;
    VkDescriptorSetLayout forwardSetLayout;

    VkDescriptorPool forwardDescriptorPool;
    VkDescriptorSet forwardDescriptorSet;

    VkCommandPool forwardCommandPool;
    VkCommandBuffer forwardCommandBuffer;

    VkDeviceMemory backwardDeviceMemory;

    VkPipeline backwardPipeline;
    VkPipelineLayout backwardPipelineLayout;
    VkDescriptorSetLayout backwardSetLayout;

    VkDescriptorPool backwardDescriptorPool;
    VkDescriptorSet backwardDescriptorSet;

    VkCommandPool backwardCommandPool;
    VkCommandBuffer backwardCommandBuffer;

public:
    virtual void forward(VkQueue& queue) = 0;
    virtual void backward(VkQueue& queue) = 0;
    virtual void backward_initialize(VkBuffer& d_out) = 0;
    virtual void forward_initialize(VkQueue& queue) = 0;

    VkBuffer& get_output(){return output;}
    VkBuffer& get_d_output(){return d_output;}

    VkDeviceMemory& get_forward_device_memory(){return forwardDeviceMemory;}
    VkDeviceMemory& get_backward_device_memory(){return backwardDeviceMemory;}
    VkBuffer& get_d_input(){return d_input;}
    uint64_t get_d_input_offset(){return backward_offsets[0];}

    virtual uint64_t get_output_offset() = 0;

    virtual uint32_t get_output_dim() = 0;
    virtual uint32_t get_input_dim() = 0;

    virtual std::vector<std::pair<VkBuffer, VkBuffer>> get_trainable_parameters() = 0;
};
#endif //VULKAN_PERCEPTRON_LAYER_H
