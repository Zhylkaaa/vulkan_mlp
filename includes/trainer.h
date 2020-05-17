//
// Created by Dima Zhylko on 17/05/2020.
//

#ifndef VULKAN_PERCEPTRON_TRAINER_H
#define VULKAN_PERCEPTRON_TRAINER_H
#include <MLP.h>

class Trainer {
protected:
    MLP mlp;
    std::string optimizer;
    VkDeVkDeviceMemory deviceMemory;
    std::vector<uint64_t> offsets;

    VkPipeline lossPipeline;
    VkPipelineLayout lossPipelineLayout;
    VkDescriptorSetLayout lossSetLayout;

    VkDescriptorPool lossDescriptorPool;
    VkDescriptorSet lossDescriptorSet;

    VkCommandPool lossCommandPool;
    VkCommandBuffer lossCommandBuffer;

    VkBuffer loss;

    VkDevice& get_device() {return device;}
    uint32_t get_queue_index() {return queueFamilyIndex;}
    VkPhysicalDevice& get_physicalDevice() {return physicalDevice;}
    std::vector<Layer*>& get_layers() {return mlp.layers;}

    virtual void train(uint32_t num_iterations, uint32_t print_every=0) = 0;
    virtual float compute_loss(const std::vector<std::vector<float>>& labels) = 0;
};

#endif //VULKAN_PERCEPTRON_TRAINER_H
