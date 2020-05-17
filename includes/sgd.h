//
// Created by @Zhylkaaa on 17/05/2020.
//

#ifndef VULKAN_PERCEPTRON_SGD_H
#define VULKAN_PERCEPTRON_SGD_H

#include <optimizer.h>
#include <vulkan_init.h>

class SGD: public Optimizer {
    VkDeviceMemory forwardDeviceMemory;

    VkPipeline forwardPipeline;
    VkPipelineLayout forwardPipelineLayout;
    VkDescriptorSetLayout forwardSetLayout;

    VkDescriptorPool forwardDescriptorPool;
    VkDescriptorSet forwardDescriptorSet;

    VkCommandPool forwardCommandPool;
    VkCommandBuffer forwardCommandBuffer;

public:
    SGD(const std::unordered_map<std::string, float>& optimizer_params);

    void init(const std::vector<std::pair<VkBuffer, VkBuffer>>& trainable_parameters) override;
    void optimize() override;
};
#endif //VULKAN_PERCEPTRON_SGD_H
