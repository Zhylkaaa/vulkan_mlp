//
// Created by Dima Zhylko on 12/05/2020.
//

#include <MLP.h>

MLP::MLP(uint32_t input_size, uint32_t batch_size, const std::vector<int> &layer_dims, const std::vector<std::string> &activations) {
    setup_vulkan(instance, debugMessenger, physicalDevice, queueFamilyIndex, device, queue);

    for(int i = 0;i<layer_dims.size();i++){
        add(layer_dims[i], activations[i], input_size, batch_size);
    }
}

void MLP::add(int layer_dim, const std::string &activation, uint32_t input_size, uint32_t batch_size) {
    if(layers.empty() && (input_size == 0 || batch_size == 0)){
        throw std::invalid_argument("first layer should specify input and batch size greater than 0");
    }

    uint32_t input_dim;

    VkBuffer input_buffer;

    if(layers.empty()){
        input_dim = input_size;
        this->batch_size = batch_size;
        this->input_size = input_size;

        createBuffer(device, queueFamilyIndex, input, batch_size, input_size);
        //createBuffer(device, queueFamilyIndex, d_input, batch_size, input_size);
        input_buffer = input;

    } else {
        input_buffer = layers[layers.size()-1]->get_output();
        input_dim = layers[layers.size()-1]->get_output_dim();
    }

    DenseLayer* d = new DenseLayer(device, queueFamilyIndex, physicalDevice, this->batch_size, input_dim, layer_dim, input_buffer);

    layers.push_back(d);

    if(activation == "id")return;

    Layer* activation_layer;

    if(activation == "relu"){
        activation_layer = new ReLULayer(device, queueFamilyIndex, physicalDevice, this->batch_size, layer_dim, d->get_output());
    } else if(activation == "softmax"){
        activation_layer = new SoftmaxLayer(device, queueFamilyIndex, physicalDevice, this->batch_size, layer_dim, d->get_output());
    } else {
        std::string error_message = "No matching activation function for " + activation;
        throw std::invalid_argument(error_message);
    }

    layers.push_back(activation_layer);
}

void MLP::forward_initialize(){
    std::vector<VkBuffer*> buffers{&input};
    allocateAndBindBuffers(device, physicalDevice, buffers, deviceMemory, offsets);

    for(Layer* layer : layers){
        layer->forward_initialize(queue);
    }
}

void MLP::forward(const std::vector<std::vector<float> > &batch) {
    char *data = nullptr;
    if(vkMapMemory(device, deviceMemory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&data)) != VK_SUCCESS){
        throw std::runtime_error("failed to map device memory");
    }
    float* batch_data = reinterpret_cast<float*>(data + offsets[0]);

    if(batch.size() != this->batch_size || batch[0].size() != this->input_size){
        throw std::invalid_argument("batch size or input dimension is wrong");
    }

    for(int i = 0;i<this->batch_size;i++){
        for(int j=0;j<this->input_size;j++){
            batch_data[i*this->input_size + j] = batch[i][j];
        }
    }

    vkUnmapMemory(device, deviceMemory);

    for(Layer* layer : layers){
        layer->forward(queue);
    }
}

MLP::MLP() {
    setup_vulkan(instance, debugMessenger, physicalDevice, queueFamilyIndex, device, queue);
}

void MLP::backward_initialize(VkBuffer& d_out) {
    layers.back()->backward_initialize(d_out);

    for(int i = layers.size()-2;i>=0;i--){
        layers[i]->backward_initialize(layers[i+1]->get_d_input());
    }
}

void MLP::backward() {
    for(int i = layers.size()-1;i>=0;i--){
        layers[i]->backward(queue);
    }
}

std::vector<std::pair<Tensor, Tensor>> MLP::get_trainable_parameters() {
    std::vector<std::pair<Tensor, Tensor>> params;

    for(Layer* layer : layers){
        std::vector<std::pair<Tensor, Tensor>> layer_params = layer->get_trainable_parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }

    return params;
}

MLP::~MLP() {

    for(Layer* layer : layers){
        delete layer;
    }

    vkFreeMemory(device, deviceMemory, nullptr);

    vkDestroyBuffer(device, input, nullptr);

    vkDestroyDevice(device, nullptr);

    DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);

    vkDestroyInstance(instance, nullptr);
}

void MLP::eval_batch(std::vector<std::vector<float>>& batch, VkCommandBuffer& evalCommandBuffer,
        VkDeviceMemory evalDeviceMemory, std::vector<uint64_t>& eval_offsets, uint32_t& correct_predictions,
        std::vector<std::vector<float>>& true_labels){

    forward(batch);
    submitTask(queue, &evalCommandBuffer);

    char *data = nullptr;
    if(vkMapMemory(device, evalDeviceMemory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&data)) != VK_SUCCESS){
        throw std::runtime_error("failed to map device memory");
    }

    uint32_t* p_labels = reinterpret_cast<uint32_t*>(data + eval_offsets[0]);

    for(uint32_t k = 0;k<batch_size;k++){
        if(true_labels[k][p_labels[k]] == 1)correct_predictions++;
    }

    vkUnmapMemory(device, evalDeviceMemory);
}

float MLP::evaluate(std::vector<std::vector<float>> &X, std::vector<std::vector<float>> &y) {
    if(X.size() != y.size() || X.empty()){
        throw std::invalid_argument("X and y should have same size > 0");
    }

    if(y[0].size() != get_output_dim()){
        throw std::invalid_argument("model's output size doesn't much label's");
    }

    uint32_t iters = X.size() / batch_size;

    uint32_t ex = 0;

    VkBuffer predicted_labels;
    createBuffer(device, queueFamilyIndex, predicted_labels, batch_size, 1, sizeof(uint32_t));

    std::vector<VkBuffer*> buffers{&predicted_labels};

    VkDeviceMemory evalDeviceMemory;
    std::vector<uint64_t> eval_offsets;
    VkDescriptorSetLayout evalSetLayout;
    VkPipelineLayout evalPipelineLayout;
    VkPipeline evalPipeline;
    VkDescriptorPool evalDescriptorPool;
    VkDescriptorSet evalDescriptorSet;
    VkCommandPool evalCommandPool;
    VkCommandBuffer evalCommandBuffer;

    struct push_const {
        uint32_t batch_size;
        uint32_t inp_dim;
    } dim{};

    dim.batch_size = batch_size;
    dim.inp_dim = get_output_dim();

    allocateAndBindBuffers(device, physicalDevice, buffers, evalDeviceMemory, eval_offsets);

    createPipelineLayout(device, 2, evalSetLayout, evalPipelineLayout, sizeof(push_const));
    createComputePipeline(device, "../shaders/argmax.comp.spv", evalPipelineLayout, evalPipeline);

    buffers.insert(buffers.begin(), &get_output());

    allocateDescriptorSet(device, buffers, evalDescriptorPool, evalSetLayout, evalDescriptorSet);
    createCommandPoolAndBuffer(device, queueFamilyIndex, evalCommandPool, evalCommandBuffer);

    recordComputePipeline(evalCommandBuffer, evalPipelineLayout, sizeof(push_const), reinterpret_cast<void*>(&dim),
                          evalPipeline, evalDescriptorSet, (dim.batch_size+31)/32, 1, 1);

    uint32_t correct_predictions = 0;

    std::vector<std::vector<float>> batch(batch_size);
    std::vector<std::vector<float>> true_labels(batch_size);

    for(uint32_t i = 0;i<iters;i++){
        for(uint32_t j=0;j<batch_size;j++){
            batch[j] = X[ex];
            true_labels[j] = y[ex];
            ex++;
        }

        eval_batch(batch, evalCommandBuffer, evalDeviceMemory, eval_offsets, correct_predictions, true_labels);
    }

    if(ex != X.size()){
        uint32_t j;
        for(j=0;j<batch_size && ex != X.size();j++){
            batch[j] = X[ex];
            true_labels[j] = y[ex];
            ex++;
        }

        for(;j<batch_size;j++){
            batch[j] = std::vector<float>(input_size, 0);
            true_labels[j] = std::vector<float>(get_output_dim(), 0);
        }

        eval_batch(batch, evalCommandBuffer, evalDeviceMemory, eval_offsets, correct_predictions, true_labels);
    }

    vkDestroyCommandPool(device, evalCommandPool, nullptr);
    vkFreeMemory(device, evalDeviceMemory, nullptr);
    vkDestroyBuffer(device, predicted_labels, nullptr);
    vkDestroyDescriptorPool(device, evalDescriptorPool, nullptr);
    vkDestroyPipeline(device, evalPipeline, nullptr);
    vkDestroyPipelineLayout(device, evalPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, evalSetLayout, nullptr);

    return static_cast<float>(correct_predictions) / X.size();
}



