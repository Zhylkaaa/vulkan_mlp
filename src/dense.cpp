//
// Created by Dima Zhylko on 12/05/2020.
//

#include <dense.h>
#include <random>

VkBuffer &DenseLayer::get_weight() {
    return weight;
}

VkBuffer &DenseLayer::get_bias() {
    return bias;
}

void DenseLayer::backward(VkQueue &queue) {

}

void DenseLayer::forward(VkQueue &queue) {
    submitTask(queue, &forwardCommandBuffer);
}

void DenseLayer::forward_initialize(VkQueue &queue) {

    std::cout<<"dense forward init"<<std::endl;

    std::vector<VkBuffer*> buffers{&weight, &bias, &output};
    allocateAndBindBuffers(device, physicalDevice, buffers, deviceMemory, offsets);

    createPipelineLayout(device, 4, forwardSetLayout, forwardPipelineLayout, sizeof(dims));
    createComputePipeline(device, "../shaders/dense.comp.spv", forwardPipelineLayout, forwardPipeline);

    buffers.insert(buffers.begin(), &input);

    allocateDescriptorSet(device, buffers, forwardDescriptorPool, forwardSetLayout, forwardDescriptorSet);
    createCommandPoolAndBuffer(device, queueFamilyIndex, forwardCommandPool, forwardCommandBuffer);

    recordComputePipeline(forwardCommandBuffer, forwardPipelineLayout, sizeof(dims), reinterpret_cast<void*>(&dim),
            forwardPipeline,forwardDescriptorSet, (dim.batch_size+15)/16, (dim.output_dim+15)/16, 1);

    // TODO: actual xavier initialization
    char* data = nullptr;
    if(vkMapMemory(device, deviceMemory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&data)) != VK_SUCCESS){
        throw std::runtime_error("failed to map device memory");
    }

    std::cout<<"obtain pointers"<<std::endl;

    float* weight = reinterpret_cast<float*>(data + offsets[0]);
    float* bias = reinterpret_cast<float*>(data + offsets[1]);

    std::cout<<"init bias"<<std::endl;
    for(int i = 0;i<dim.output_dim;i++){
        bias[i] = 0;
    }
    std::cout<<"end init bias"<<std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0, 1.0);

    std::cout<<"init weight"<<std::endl;
    for(int i = 0;i<dim.inp_dim;i++){
        for(int j = 0;j<dim.output_dim;j++){
            weight[i*dim.output_dim + j] = scale * dis(gen);
        }
    }
    std::cout<<"end init weight"<<std::endl;
    vkUnmapMemory(device, deviceMemory);

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

    createBuffer(device, queueFamilyIndex, weight, dim.inp_dim, dim.output_dim);
    createBuffer(device, queueFamilyIndex, bias, dim.output_dim, 1);
    createBuffer(device, queueFamilyIndex, output, dim.batch_size, dim.output_dim);
}
