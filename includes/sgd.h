//
// Created by @Zhylkaaa on 17/05/2020.
//

#ifndef VULKAN_PERCEPTRON_SGD_H
#define VULKAN_PERCEPTRON_SGD_H

#include <optimizer.h>
#include <vulkan_init.h>
#include <iostream>

class SGD: public Optimizer {

    struct push_constant {
        float lr;
        Tensor::dims dim;
    };

    std::vector<push_constant> pushConstant;
    std::vector<VkPipeline> optimizePipeline;
    std::vector<VkPipelineLayout> optimizePipelineLayout;
    std::vector<VkDescriptorSetLayout> optimizeSetLayout;

    VkDescriptorPool optimizeDescriptorPool;
    std::vector<VkDescriptorSet> optimizeDescriptorSet;

    VkCommandPool optimizeCommandPool;
    std::vector<VkCommandBuffer> optimizeCommandBuffer;

    VkDevice device;
public:
    SGD(const std::unordered_map<std::string, float>& optimizer_params);

    void init(const VkDevice& device, uint32_t queueFamilyIndex, std::vector<std::pair<Tensor, Tensor>> trainable_parameters) override;
    void optimize(VkQueue& queue) override;

    ~SGD();
};
#endif //VULKAN_PERCEPTRON_SGD_H
