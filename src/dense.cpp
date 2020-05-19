//
// Created by Dima Zhylko on 12/05/2020.
//

#include <dense.h>
#include <random>

void DenseLayer::forward(VkQueue &queue) {
    submitTask(queue, &forwardCommandBuffer);
}

void DenseLayer::forward_initialize(VkQueue &queue) {

    std::vector<VkBuffer*> buffers{&weight.get_buffer(), &bias.get_buffer(), &output};
    allocateAndBindBuffers(device, physicalDevice, buffers, forwardDeviceMemory, forward_offsets);

    createPipelineLayout(device, 4, forwardSetLayout, forwardPipelineLayout, sizeof(dims));
    createComputePipeline(device, "../shaders/dense.comp.spv", forwardPipelineLayout, forwardPipeline);

    buffers.insert(buffers.begin(), &input);

    allocateDescriptorSet(device, buffers, forwardDescriptorPool, forwardSetLayout, forwardDescriptorSet);
    createCommandPoolAndBuffer(device, queueFamilyIndex, forwardCommandPool, forwardCommandBuffer);

    recordComputePipeline(forwardCommandBuffer, forwardPipelineLayout, sizeof(dims), reinterpret_cast<void*>(&dim),
            forwardPipeline,forwardDescriptorSet, (dim.batch_size+15)/16, (dim.output_dim+15)/16, 1);

    // TODO: actual He-et-al initialization
    char* data = nullptr;
    if(vkMapMemory(device, forwardDeviceMemory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&data)) != VK_SUCCESS){
        throw std::runtime_error("failed to map device memory");
    }

    if(initializer == "He-et-al"){
        float* weight = reinterpret_cast<float*>(data + forward_offsets[0]);
        float* bias = reinterpret_cast<float*>(data + forward_offsets[1]);

        for(int i = 0;i<dim.output_dim;i++){
            bias[i] = 0;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(0.0, 2.0/dim.inp_dim);

        for(int i = 0;i<dim.inp_dim;i++){
            for(int j = 0;j<dim.output_dim;j++){
                weight[i*dim.output_dim + j] = dis(gen);
            }
        }
    } else {
        throw std::invalid_argument("unknown initializer");
    }
    vkUnmapMemory(device, forwardDeviceMemory);

}

DenseLayer::DenseLayer(VkDevice device, uint32_t queueFamilyIndex, VkPhysicalDevice physicalDevice,
        int batch_size, int input_dim, int output_dim, VkBuffer input, float scale, const std::string& initializer) {
    this->scale = scale;

    this->input = input;
    this->initializer = initializer;
    dim.batch_size = batch_size;
    dim.inp_dim = input_dim;
    dim.output_dim = output_dim;

    this->device = device;
    this->queueFamilyIndex = queueFamilyIndex;
    this->physicalDevice = physicalDevice;

    createBuffer(device, queueFamilyIndex, weight.get_buffer(), dim.inp_dim, dim.output_dim);
    weight.set_height(dim.inp_dim);
    weight.set_width(dim.output_dim);

    createBuffer(device, queueFamilyIndex, bias.get_buffer(), dim.output_dim, 1);
    bias.set_height(dim.output_dim);
    bias.set_width(1);

    createBuffer(device, queueFamilyIndex, output, dim.batch_size, dim.output_dim);
}

void DenseLayer::backward(VkQueue &queue) {
    submitTask(queue, &backwardCommandBuffer, false);
    submitTask(queue, &backwardWeightCommandBuffer, false);
    submitTask(queue, &backwardBiasCommandBuffer);
}

void DenseLayer::backward_initialize(VkBuffer &d_out) {
    d_output = d_out;

    createBuffer(device, queueFamilyIndex, d_input, dim.batch_size, dim.inp_dim);
    createBuffer(device, queueFamilyIndex, d_weight.get_buffer(), dim.inp_dim, dim.output_dim);
    d_weight.set_height(dim.inp_dim);
    d_weight.set_width(dim.output_dim);

    createBuffer(device, queueFamilyIndex, d_bias.get_buffer(), dim.output_dim, 1);
    d_bias.set_height(dim.output_dim);
    d_bias.set_width(1);

    std::vector<VkBuffer*> buffers{&d_input, &d_weight.get_buffer(), &d_bias.get_buffer()};

    allocateAndBindBuffers(device, physicalDevice, buffers, backwardDeviceMemory, backward_offsets);

    createPipelineLayout(device, 7, backwardSetLayout, backwardPipelineLayout, sizeof(dims));
    createComputePipeline(device, "../shaders/d_dense.comp.spv", backwardPipelineLayout, backwardPipeline);
    createComputePipeline(device, "../shaders/d_dense_w.comp.spv", backwardPipelineLayout, backwardWeightPipeline);
    createComputePipeline(device, "../shaders/d_dense_b.comp.spv", backwardPipelineLayout, backwardBiasPipeline);

    buffers.insert(buffers.begin(), &bias.get_buffer());
    buffers.insert(buffers.begin(), &weight.get_buffer());
    buffers.insert(buffers.begin(), &input);
    buffers.push_back(&d_output);

    allocateDescriptorSet(device, buffers, backwardDescriptorPool, backwardSetLayout, backwardDescriptorSet);
    createCommandPoolAndBuffer(device, queueFamilyIndex, backwardCommandPool, backwardCommandBuffer);
    createCommandPoolAndBuffer(device, queueFamilyIndex, backwardWeightCommandPool, backwardWeightCommandBuffer);
    createCommandPoolAndBuffer(device, queueFamilyIndex, backwardBiasCommandPool, backwardBiasCommandBuffer);

    recordComputePipeline(backwardCommandBuffer, backwardPipelineLayout, sizeof(dims), reinterpret_cast<void*>(&dim),
                          backwardPipeline,backwardDescriptorSet, (dim.batch_size+15)/16, (dim.inp_dim+15)/16, 1);

    recordComputePipeline(backwardWeightCommandBuffer, backwardPipelineLayout, sizeof(dims), reinterpret_cast<void*>(&dim),
                          backwardWeightPipeline,backwardDescriptorSet, (dim.inp_dim+15)/16, (dim.output_dim+15)/16, 1);

    recordComputePipeline(backwardBiasCommandBuffer, backwardPipelineLayout, sizeof(dims), reinterpret_cast<void*>(&dim),
                          backwardBiasPipeline,backwardDescriptorSet, (dim.output_dim+63)/64, 1, 1);
    backward_initialized = true;
}

std::vector<std::pair<Tensor, Tensor>> DenseLayer::get_trainable_parameters() {
    std::vector<std::pair<Tensor, Tensor>> params{
            {weight, d_weight},
            {bias, d_bias}
    };
    return params;
}

Tensor &DenseLayer::get_bias() {
    return bias;
}

Tensor &DenseLayer::get_weight() {
    return weight;
}

DenseLayer::~DenseLayer() {
    vkDestroyCommandPool(device, forwardCommandPool, nullptr);

    vkFreeMemory(device, forwardDeviceMemory, nullptr);

    vkDestroyBuffer(device, output, nullptr);
    vkDestroyBuffer(device, weight.get_buffer(), nullptr);
    vkDestroyBuffer(device, bias.get_buffer(), nullptr);

    vkDestroyDescriptorPool(device, forwardDescriptorPool, nullptr);

    vkDestroyPipeline(device, forwardPipeline, nullptr);

    vkDestroyPipelineLayout(device, forwardPipelineLayout, nullptr);

    vkDestroyDescriptorSetLayout(device, forwardSetLayout, nullptr);

    if(backward_initialized) {
        vkDestroyCommandPool(device, backwardCommandPool, nullptr);
        vkDestroyCommandPool(device, backwardBiasCommandPool, nullptr);
        vkDestroyCommandPool(device, backwardWeightCommandPool, nullptr);
        vkFreeMemory(device, backwardDeviceMemory, nullptr);
        vkDestroyBuffer(device, d_input, nullptr);
        vkDestroyBuffer(device, d_weight.get_buffer(), nullptr);
        vkDestroyBuffer(device, d_bias.get_buffer(), nullptr);
        vkDestroyDescriptorPool(device, backwardDescriptorPool, nullptr);
        vkDestroyPipeline(device, backwardPipeline, nullptr);
        vkDestroyPipeline(device, backwardWeightPipeline, nullptr);
        vkDestroyPipeline(device, backwardBiasPipeline, nullptr);
        vkDestroyPipelineLayout(device, backwardPipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, backwardSetLayout, nullptr);
    }
}
