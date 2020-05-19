//
// Created by Dima Zhylko on 17/05/2020.
//

#ifndef VULKAN_PERCEPTRON_TRAINER_H
#define VULKAN_PERCEPTRON_TRAINER_H

#include <MLP.h>
#include <optimizer.h>

class Trainer {
protected:
    MLP* mlp;
    std::string optimizer;
    VkDeviceMemory deviceMemory;
    std::vector<uint64_t> offsets;

    Optimizer* parameters_optimizer;

    VkPipeline lossPipeline;
    VkPipelineLayout lossPipelineLayout;
    VkDescriptorSetLayout lossSetLayout;

    VkDescriptorPool lossDescriptorPool;
    VkDescriptorSet lossDescriptorSet;

    VkCommandPool lossCommandPool;
    VkCommandBuffer lossCommandBuffer;

    VkBuffer loss;

    VkDevice& get_device() {return mlp->device;}
    uint32_t get_queue_index() {return mlp->queueFamilyIndex;}
    VkPhysicalDevice& get_physicalDevice() {return mlp->physicalDevice;}
    std::vector<Layer*>& get_layers() {return mlp->layers;}
    VkQueue& get_queue() {return mlp->queue;}

    virtual void train(uint32_t num_iterations, uint32_t print_every) = 0;
    virtual void train(uint32_t num_iterations, std::vector<float>& loss_history, uint32_t print_every) = 0;

    virtual float compute_loss(const std::vector<std::vector<float>>& labels) = 0;

    virtual ~Trainer() = default;
};

#endif //VULKAN_PERCEPTRON_TRAINER_H
