//
// Created by Dima Zhylko on 17/05/2020.
//

#include <classification_trainer.h>

ClassificationTrainer::ClassificationTrainer(MLP mlp,
                                             std::vector<example> &dataset,
                                             const std::unordered_map<std::string, float>& optimizer_params,
                                             const std::string& optimizer) {

    if(dynamic_cast<SoftmaxLayer*>(*get_layers().end()) == nullptr){
        std::runtime_error("Model last layer should be of type softmax, do this by specifying \"softmax\" as activation");
    }

    this->mlp = mlp;
    this->dataset = dataset;

    if(dataset.empty())throw std::invalid_argument("dataset should contain data :)");
    if(dataset[0].y.size() != mlp.get_output_dim())throw std::invalid_argument("output dimension should match dataset label dimension");

    createBuffer(get_device(), get_queue_index(), labels, mlp.get_batch_size(), mlp.get_output_dim());

    std::vector<VkBuffer*> buffers{&labels};
    allocateAndBindBuffers(get_device(), get_physicalDevice(), buffers, deviceMemory, offsets);

    mlp.backward_initialize(labels);

    if(optimizer == "sgd"){
        parameters_optimizer = new SGD(optimizer_params);
        parameters_optimizer->init(mlp.get_trainable_parameters());
    } else {
        throw std::invalid_argument("unknown optimizer");
    }
}

void inline create_batch(std::vector<example>& batch, std::mt19937& gen, std::uniform_int_distribution<int>& distribution,
        std::vector<example>& dataset, uint32_t batch_size){
    for(uint32_t i=0;i<batch_size; i++){
        batch[i] = dataset[distribution(gen)];
    }
}

void ClassificationTrainer::train(uint32_t num_iterations, uint32_t print_every) {
    std::vector<example> batch(mlp.get_batch_size());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(0,dataset.size()-1);

    for(int i=0;i<num_iterations;i++){
        create_batch(batch, gen, distribution, dataset, mlp.get_batch_size());

        std::vector<std::vector<float>> x_batch(mlp.get_batch_size(), std::vector<float>(dataset[0].x.size()));
        std::vector<std::vector<float>> y_batch(mlp.get_batch_size(), std::vector<float>(dataset[0].y.size()));

        uint32_t j = 0;
        for(const example& e : batch){
            x_batch[j] = e.x;
            y_batch[j] = e.y;
            j++;
        }

        mlp.forward(x_batch);

        char* data = nullptr;
        if(vkMapMemory(get_device(), deviceMemory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&data)) != VK_SUCCESS){
            throw std::runtime_error("failed to map device memory");
        }

        float* labels = reinterpret_cast<float*>(data + offsets[0]);

        for(j=0;j<mlp.get_batch_size();j++){
            std::memcpy(&labels[j*mlp.get_output_dim()], y_batch[j].data(), y_batch[j].size()*sizeof(float));
        }
        vkUnmapMemory(get_device(), deviceMemory);

        mlp.backward();

        parameters_optimizer->optimize();

        if((print_every != 0 && i % print_every == 0) || i == num_iterations-1){
            std::cout<<"step: "<<i<<"loss: "<<compute_loss(y_batch);
        }
    }
}

float ClassificationTrainer::compute_loss(const std::vector<std::vector<float>> &labels) {
    float result_loss = 0;

    char* data = nullptr;
    if(vkMapMemory(get_device(), mlp.get_output_memory(), 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&data)) != VK_SUCCESS){
        throw std::runtime_error("failed to map device memory");
    }

    float* predictions = reinterpret_cast<float*>(data + mlp.get_output_offset());

    for(uint32_t i = 0;i<labels.size();i++){
        for(uint32_t j = 0;j<labels[0].size();j++){
            if(labels[i][j] == 1){
                result_loss += std::log(predictions[i*labels[0].size() + j] + static_cast<float>(1e-10));
            }
        }
    }
    vkUnmapMemory(get_device(), mlp.get_output_memory());
    return -result_loss;
}
