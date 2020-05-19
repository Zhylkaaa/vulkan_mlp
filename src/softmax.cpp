//
// Created by Dima Zhylko on 15/05/2020.
//

#include <softmax.h>

// TODO: create abstract class for activations

SoftmaxLayer::SoftmaxLayer(VkDevice device, uint32_t queueFamilyIndex, VkPhysicalDevice physicalDevice, int batch_size,
                           int input_dim, VkBuffer input) {
    this->device = device;
    this->queueFamilyIndex = queueFamilyIndex;
    this->physicalDevice = physicalDevice;
    dim.batch_size = batch_size;
    dim.inp_dim = input_dim;

    this->input = input;

    createBuffer(device, queueFamilyIndex, output, dim.batch_size, dim.inp_dim);
}


void SoftmaxLayer::forward(VkQueue &queue) {
    submitTask(queue, &forwardCommandBuffer);
}

void SoftmaxLayer::forward_initialize(VkQueue &queue) {
    std::vector<VkBuffer*> buffers{&output};
    allocateAndBindBuffers(device, physicalDevice, buffers, forwardDeviceMemory, forward_offsets);

    createPipelineLayout(device, 2, forwardSetLayout, forwardPipelineLayout, sizeof(dims));
    createComputePipeline(device, "../shaders/softmax.comp.spv", forwardPipelineLayout, forwardPipeline);

    buffers.insert(buffers.begin(), &input);

    allocateDescriptorSet(device, buffers, forwardDescriptorPool, forwardSetLayout, forwardDescriptorSet);
    createCommandPoolAndBuffer(device, queueFamilyIndex, forwardCommandPool, forwardCommandBuffer);

    recordComputePipeline(forwardCommandBuffer, forwardPipelineLayout, sizeof(dims), reinterpret_cast<void*>(&dim),
                          forwardPipeline,forwardDescriptorSet, (dim.batch_size+63)/64, 1, 1);

}

void SoftmaxLayer::backward(VkQueue &queue) {
    submitTask(queue, &backwardCommandBuffer);
}

void SoftmaxLayer::backward_initialize(VkBuffer &d_out) {
    d_output = d_out;

    createBuffer(device, queueFamilyIndex, d_input, dim.batch_size, dim.inp_dim);

    std::vector<VkBuffer*> buffers{&d_input};
    allocateAndBindBuffers(device, physicalDevice, buffers, backwardDeviceMemory, backward_offsets);

    createPipelineLayout(device, 3, backwardSetLayout, backwardPipelineLayout, sizeof(dims));
    createComputePipeline(device, "../shaders/d_softmax.comp.spv", backwardPipelineLayout, backwardPipeline);

    buffers.insert(buffers.begin(), &output);
    buffers.push_back(&d_output);

    allocateDescriptorSet(device, buffers, backwardDescriptorPool, backwardSetLayout, backwardDescriptorSet);
    createCommandPoolAndBuffer(device, queueFamilyIndex, backwardCommandPool, backwardCommandBuffer);

    recordComputePipeline(backwardCommandBuffer, backwardPipelineLayout, sizeof(dims), reinterpret_cast<void*>(&dim),
                          backwardPipeline,backwardDescriptorSet, (dim.batch_size+15)/16, (dim.inp_dim+15)/16, 1);
    backward_initialized = true;
}

std::vector<std::pair<Tensor, Tensor>> SoftmaxLayer::get_trainable_parameters() {
    return std::vector<std::pair<Tensor, Tensor>>();
}

SoftmaxLayer::~SoftmaxLayer() {
    vkDestroyCommandPool(device, forwardCommandPool, nullptr);

    vkFreeMemory(device, forwardDeviceMemory, nullptr);

    vkDestroyBuffer(device, output, nullptr);

    vkDestroyDescriptorPool(device, forwardDescriptorPool, nullptr);

    vkDestroyPipeline(device, forwardPipeline, nullptr);

    vkDestroyPipelineLayout(device, forwardPipelineLayout, nullptr);

    vkDestroyDescriptorSetLayout(device, forwardSetLayout, nullptr);

    if(backward_initialized){
        vkDestroyCommandPool(device, backwardCommandPool, nullptr);
        vkFreeMemory(device, backwardDeviceMemory, nullptr);
        vkDestroyBuffer(device, d_input, nullptr);
        vkDestroyDescriptorPool(device, backwardDescriptorPool, nullptr);
        vkDestroyPipeline(device, backwardPipeline, nullptr);
        vkDestroyPipelineLayout(device, backwardPipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, backwardSetLayout, nullptr);
    }
}
