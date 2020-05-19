#include <iostream>
#include <MLP.h>
#include <string>
#include <vulkan_init.h>
#include <classification_trainer.h>
#include <fstream>

int main() {

    std::vector<int> layers{10};
    std::vector<std::string> activations{"softmax"};
    MLP mlp = MLP(784, 32, layers, activations);

    mlp.forward_initialize();

    std::vector<example> train_dataset;

    std::vector<std::vector<float>> val_x;
    std::vector<std::vector<float>> val_y;

    std::ifstream train_image_input("../train_MNIST_images.txt");
    std::ifstream val_image_input("../val_MNIST_images.txt", std::ios::in);

    std::ifstream train_label_input("../train_MNIST_labels.txt", std::ios::in);
    std::ifstream val_label_input("../val_MNIST_labels.txt", std::ios::in);

    if(!train_image_input.is_open() || !val_image_input.is_open() || !train_label_input.is_open() ||
    !val_label_input.is_open())throw std::runtime_error("can't read training data");

    for(int i = 0;i<20000;i++){
        std::vector<float> train_x(784);
        std::vector<float> v_x(784);

        std::vector<float> train_y(10);
        std::vector<float> v_y(10);
        for(int j = 0;j<784;j++){
            train_image_input>>train_x[j];
            if(i<10000)val_image_input>>v_x[j];
        }

        for(int j = 0;j<10;j++){
            train_label_input>>train_y[j];
            if(i<10000)val_label_input>>v_y[j];
        }

        example train_e{.x=train_x, .y=train_y};

        train_dataset.push_back(train_e);
        if(i<10000){
            val_x.push_back(v_x);
            val_y.push_back(v_y);
        }
    }

    std::cout<<"accuracy before training: "<<mlp.evaluate(val_x, val_y)<<std::endl;

    std::unordered_map<std::string, float> params;
    params["learning_rate"] = 0.3;

    ClassificationTrainer trainer = ClassificationTrainer(&mlp, train_dataset, params);

    std::vector<float> loss_history;

    trainer.train(1000, loss_history, 1);

    std::cout<<"accuracy after training: "<<mlp.evaluate(val_x, val_y)<<std::endl;

    return 0;
}