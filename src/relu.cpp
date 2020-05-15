//
// Created by Dima Zhylko on 15/05/2020.
//

#include <relu.h>

ReLULayer::ReLULayer(VkDevice device, uint32_t queueFamilyIndex, VkPhysicalDevice physicalDevice, int batch_size,
                     int input_dim, VkBuffer input) {
    this->device = device;
    this->queueFamilyIndex = queueFamilyIndex;
    this->physicalDevice = physicalDevice;
    dim.batch_size = batch_size;
    dim.inp_dim = input_dim;

    this->input = input;

    createBuffer(device, queueFamilyIndex, output, dim.batch_size, dim.inp_dim);
}

void ReLULayer::forward_initialize(VkQueue &queue) {
    std::vector<VkBuffer*> buffers{&output};
    allocateAndBindBuffers(device, physicalDevice, buffers, deviceMemory, offsets);

    createPipelineLayout(device, 2, forwardSetLayout, forwardPipelineLayout, sizeof(dims));
    createComputePipeline(device, "../shaders/relu.comp.spv", forwardPipelineLayout, forwardPipeline);

    buffers.insert(buffers.begin(), &input);

    allocateDescriptorSet(device, buffers, forwardDescriptorPool, forwardSetLayout, forwardDescriptorSet);
    createCommandPoolAndBuffer(device, queueFamilyIndex, forwardCommandPool, forwardCommandBuffer);

    recordComputePipeline(forwardCommandBuffer, forwardPipelineLayout, sizeof(dims), reinterpret_cast<void*>(&dim),
                          forwardPipeline,forwardDescriptorSet, (dim.batch_size+15)/16, (dim.inp_dim+15)/16, 1);

}

void ReLULayer::forward(VkQueue &queue) {
    submitTask(queue, &forwardCommandBuffer);
}

void ReLULayer::backward(VkQueue &queue) {

}
