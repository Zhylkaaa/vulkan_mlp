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
        std::cout<<"forward_initialize:"<<std::endl;
        layer->forward_initialize(queue);
    }
}

void MLP::forward(const std::vector<std::vector<float> > &batch) {
    char *data = nullptr;
    if(vkMapMemory(device, deviceMemory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&data)) != VK_SUCCESS){
        throw std::runtime_error("failed to map device memory");
    }
    std::cout<<"batch_init:"<<std::endl;
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
        std::cout<<"forward:"<<std::endl;
        layer->forward(queue);
    }

#ifndef NDEBUG
    std::cout<<"output is:"<<std::endl;

    int n = layers.size();
    data = nullptr;
    if(vkMapMemory(device, layers[n-1]->get_device_memory(), 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&data)) != VK_SUCCESS){
        throw std::runtime_error("failed to map device memory");
    }

    float* output = reinterpret_cast<float*>(data + layers[n-1]->get_output_offset());

    for(int i = 0;i<this->batch_size;i++){
        for(int j = 0;j<layers[n-1]->get_output_dim();j++){
            std::cout<<output[i*layers[n-1]->get_output_dim() + j]<<" ";
        }
        std::cout<<std::endl;
    }
    vkUnmapMemory(device, layers[n-1]->get_device_memory());
#endif
}

MLP::MLP() {
    setup_vulkan(instance, debugMessenger, physicalDevice, queueFamilyIndex, device, queue);
}



