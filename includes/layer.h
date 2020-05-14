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
    std::vector<uint64_t> offsets{};
    VkBuffer d_input;
    VkBuffer d_output;

    VkDevice* device;
    uint32_t queueFamilyIndex;
    VkPhysicalDevice* physicalDevice;

    VkDeviceMemory deviceMemory;

    VkPipeline forwardPipeline;
    VkPipelineLayout forwardPipelineLayout;
    VkDescriptorSetLayout forwardSetLayout;

    VkDescriptorPool forwardDescriptorPool;
    VkDescriptorSet forwardDescriptorSet;

    VkCommandPool forwardCommandPool;
    VkCommandBuffer forwardCommandBuffer;

public:
    virtual void forward(VkQueue& queue) = 0;
    virtual void backward(VkQueue& queue) = 0;
    virtual void forward_initialize(VkQueue& queue) = 0;

    VkBuffer& get_output(){return output;}
    VkDeviceMemory& get_device_memory(){return deviceMemory;}

    virtual uint64_t get_output_offset() = 0;

    virtual uint32_t get_output_dim() = 0;
};
#endif //VULKAN_PERCEPTRON_LAYER_H
