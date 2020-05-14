//
// Created by Dima Zhylko on 14/05/2020.
//

#ifndef VULKAN_PERCEPTRON_VULKAN_INIT_H
#define VULKAN_PERCEPTRON_VULKAN_INIT_H

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

void setup_vulkan(VkInstance& instance, VkDebugUtilsMessengerEXT& debugMessenger, VkPhysicalDevice& physicalDevice,
                  uint32_t& queueFamilyIndex, VkDevice& device, VkQueue& queue);

void createBuffer(const VkDevice& device, uint32_t queueFamilyIndex, VkBuffer& buffer,
                  uint32_t n, uint32_t m, uint64_t elem_size=sizeof(float));

void allocateAndBindBuffers(const VkDevice& device, const VkPhysicalDevice& physicalDevice, std::vector<VkBuffer*>& buffers,
                            VkDeviceMemory& memory, std::vector<uint64_t>& offsets);

void createPipelineLayout(const VkDevice& device, uint32_t bindingsCount, VkDescriptorSetLayout& setLayout,
                          VkPipelineLayout& pipelineLayout, uint32_t push_constant_size);

void createComputePipeline(const VkDevice& device, const std::string& shaderName, const VkPipelineLayout& pipelineLayout,
                           VkPipeline& pipeline, const std::string& entry_point="main");

void allocateDescriptorSet(const VkDevice& device, std::vector<VkBuffer*>& buffers,
                           VkDescriptorPool& descriptorPool, const VkDescriptorSetLayout &setLayout, VkDescriptorSet& descriptorSet);

void createCommandPoolAndBuffer(const VkDevice& device, uint32_t queueFamilyIndex,
                                VkCommandPool& commandPool, VkCommandBuffer& commandBuffer, VkCommandPoolCreateFlags flags=0);

void recordComputePipeline(VkCommandBuffer& commandBuffer, const VkPipelineLayout& pipelineLayout,
                           uint32_t push_constant_size, void* push_constant_vals, const VkPipeline& pipeline,
                           VkDescriptorSet& descriptorSet, uint32_t x_group, uint32_t y_group, uint32_t z_group,
                           VkCommandBufferUsageFlags flags=0);
#endif //VULKAN_PERCEPTRON_VULKAN_INIT_H
