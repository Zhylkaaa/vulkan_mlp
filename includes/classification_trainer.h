//
// Created by Dima Zhylko on 17/05/2020.
//

#ifndef VULKAN_PERCEPTRON_CLASSIFICATION_TRAINER_H
#define VULKAN_PERCEPTRON_CLASSIFICATION_TRAINER_H

#include <trainer.h>
#include <vector>
#include <string>
#include <vulkan_init.h>
#include <random>
#include <sgd.h>
#include <optimizer.h>

struct example {
    std::vector<float> x;
    std::vector<float> y;
};

class ClassificationTrainer: public Trainer {
    VkBuffer labels;
    std::vector<example> dataset;

    struct dims {
        uint32_t batch_size;
        uint32_t output_dim;
    } dim;
public:

    ClassificationTrainer(MLP* mlp, std::vector<example> &dataset, const std::unordered_map<std::string, float>& optimizer_params,
                          const std::string& optimizer="sgd");

    void train(uint32_t num_iterations, uint32_t print_every=0) override;

    float compute_loss(const std::vector<std::vector<float>>& labels) override;

    ~ClassificationTrainer();
};

#endif //VULKAN_PERCEPTRON_CLASSIFICATION_TRAINER_H