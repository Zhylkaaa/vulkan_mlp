//
// Created by Dima Zhylko on 17/05/2020.
//

#include <classification_trainer.h>

ClassificationTrainer::ClassificationTrainer(MLP* mlp,
                                             std::vector<example> &dataset,
                                             const std::unordered_map<std::string, float>& optimizer_params,
                                             const std::string& optimizer) {
    this->mlp = mlp;

    if(get_layers().empty())throw std::invalid_argument("model should contain some layers");

    this->dataset = dataset;

    if(dataset.empty())throw std::invalid_argument("dataset should contain data :)");
    if(dataset[0].y.size() != mlp->get_output_dim())throw std::invalid_argument("output dimension should match dataset label dimension");

    createBuffer(get_device(), get_queue_index(), labels, mlp->get_batch_size(), mlp->get_output_dim());

    std::vector<VkBuffer*> buffers{&labels};
    allocateAndBindBuffers(get_device(), get_physicalDevice(), buffers, deviceMemory, offsets);

    mlp->backward_initialize(labels);

    if(optimizer == "sgd"){
        parameters_optimizer = new SGD(optimizer_params);
        parameters_optimizer->init(get_device(), get_queue_index(), mlp->get_trainable_parameters());
    } else {
        throw std::invalid_argument("unknown optimizer");
    }
}

void inline create_batch(std::vector<example>& batch, uint32_t& iterator,
        std::vector<example>& dataset, uint32_t batch_size){
    for(uint32_t k = 0;k<batch_size;k++){
        batch[k] = dataset[iterator++];
        iterator %= dataset.size();
    }
}

void ClassificationTrainer::train(uint32_t num_iterations, uint32_t print_every) {
    std::vector<float> tmp_loss_history;
    train(num_iterations, tmp_loss_history, print_every);
}

void ClassificationTrainer::train(uint32_t num_iterations, std::vector<float> &loss_history, uint32_t print_every) {
    std::vector<example> batch(mlp->get_batch_size());

    uint32_t iterator = 0;

    for(int i=0;i<num_iterations;i++){
        create_batch(batch, iterator, dataset, mlp->get_batch_size());

        std::vector<std::vector<float>> x_batch(mlp->get_batch_size(), std::vector<float>(dataset[0].x.size()));
        std::vector<std::vector<float>> y_batch(mlp->get_batch_size(), std::vector<float>(dataset[0].y.size()));

        uint32_t j = 0;
        for(const example& e : batch){
            x_batch[j] = e.x;
            y_batch[j] = e.y;
            j++;
        }

        mlp->forward(x_batch);

        char* data = nullptr;
        if(vkMapMemory(get_device(), deviceMemory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&data)) != VK_SUCCESS){
            throw std::runtime_error("failed to map device memory");
        }

        float* labels = reinterpret_cast<float*>(data + offsets[0]);

        for(j=0;j<mlp->get_batch_size();j++){
            for(uint32_t k=0;k<mlp->get_output_dim();k++){
                labels[j*mlp->get_output_dim() + k] = y_batch[j][k];
            }
        }

        vkUnmapMemory(get_device(), deviceMemory);

        mlp->backward();

        parameters_optimizer->optimize(get_queue());

        if((print_every != 0 && i % print_every == 0) || i == num_iterations-1){
            float l = compute_loss(y_batch);
            std::cout<<"step: "<<i<<" loss: "<<l<<std::endl;
            loss_history.push_back(l);
        }
    }
}

float ClassificationTrainer::compute_loss(const std::vector<std::vector<float>> &labels) {
    float result_loss = 0;

    char* data = nullptr;
    if(vkMapMemory(get_device(), mlp->get_output_memory(), 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&data)) != VK_SUCCESS){
        throw std::runtime_error("failed to map device memory");
    }

    float* predictions = reinterpret_cast<float*>(data + mlp->get_output_offset());

    for(uint32_t i = 0;i<labels.size();i++){
        for(uint32_t j = 0;j<labels[0].size();j++){
            if(labels[i][j] == 1){
                result_loss += std::log(predictions[i*labels[0].size() + j] + static_cast<float>(1e-10));
            }
        }
    }
    vkUnmapMemory(get_device(), mlp->get_output_memory());
    return -result_loss;
}

ClassificationTrainer::~ClassificationTrainer() {
    delete parameters_optimizer;

    vkFreeMemory(get_device(), deviceMemory, nullptr);

    vkDestroyBuffer(get_device(), labels, nullptr);
}
