//
// Created by @Zhylkaaa on 17/05/2020.
//

#include <sgd.h>

SGD::SGD(const std::unordered_map<std::string, float> &optimizer_params) {
    allowed_params.insert("learning_rate");
    set_parameters(optimizer_params);
}

void SGD::init(const VkDevice& device, uint32_t queueFamilyIndex, std::vector<std::pair<Tensor, Tensor>> trainable_parameters) {
    uint32_t n = trainable_parameters.size();

    optimizeSetLayout.resize(n);
    optimizePipelineLayout.resize(n);
    optimizePipeline.resize(n);
    pushConstant.resize(n);
    optimizeDescriptorSet.resize(n);
    optimizeCommandBuffer.resize(n);

    std::vector<std::vector<VkBuffer*>> parameter_buffers;

    for(uint32_t i = 0;i<n;i++){
        createPipelineLayout(device, 2, optimizeSetLayout[i], optimizePipelineLayout[i],
                             sizeof(push_constant));
        createComputePipeline(device, "../shaders/sgd.comp.spv", optimizePipelineLayout[i], optimizePipeline[i]);

        std::vector<VkBuffer*> buffers{&trainable_parameters[i].first.get_buffer(), &trainable_parameters[i].second.get_buffer()};
        parameter_buffers.push_back(buffers);
    }

    allocateDescriptorSet(device, parameter_buffers, optimizeDescriptorPool, optimizeSetLayout, optimizeDescriptorSet);

    createCommandPoolAndBuffer(device, queueFamilyIndex, optimizeCommandPool, optimizeCommandBuffer);

    for(uint32_t i = 0;i<n;i++){
        uint32_t x_group_size = (trainable_parameters[i].first.get_height()+15) / 16;
        uint32_t y_group_size = (trainable_parameters[i].first.get_width()+15) / 16;

        if(trainable_parameters[i].first.get_width() == 1){
            x_group_size = (x_group_size + 15) / 16;
        }

        pushConstant[i].lr = optimizer_params["learning_rate"];
        pushConstant[i].height = trainable_parameters[i].first.get_dims().height;
        pushConstant[i].width = trainable_parameters[i].first.get_dims().width;

        recordComputePipeline(optimizeCommandBuffer[i], optimizePipelineLayout[i], sizeof(push_constant),
                reinterpret_cast<void*>(&pushConstant[i]), optimizePipeline[i],
                optimizeDescriptorSet[i], x_group_size, y_group_size, 1);
    }

    this->device = device;
}

void SGD::optimize(VkQueue& queue) {
    for(VkCommandBuffer &commandBuffer : optimizeCommandBuffer){
        submitTask(queue, &commandBuffer, false);
    }

    vkQueueWaitIdle(queue);
}

SGD::~SGD() {
    vkDestroyCommandPool(device, optimizeCommandPool, nullptr);
    vkDestroyDescriptorPool(device, optimizeDescriptorPool, nullptr);
    for(int i = 0;i<optimizePipeline.size();i++){
        vkDestroyPipeline(device, optimizePipeline[i], nullptr);
        vkDestroyPipelineLayout(device, optimizePipelineLayout[i], nullptr);
        vkDestroyDescriptorSetLayout(device, optimizeSetLayout[i], nullptr);
    }
}
