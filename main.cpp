#include <iostream>
#include <MLP.h>
#include <string>
#include <vulkan_init.h>
#include <classification_trainer.h>
#include <fstream>
#include <sstream>

int find_arg(char **argv, int argc, const std::string& arg_name){
    for(int i = 0;i<argc;i++){
        if(std::string(argv[i]).find(arg_name) != std::string::npos){
            return i;
        }
    }
    return -1;
}

void get_dataset(char **argv, int argc, std::string &data_path, std::string &labels_path, bool is_train=false) {
    std::string data_arg;
    std::string labels_arg;

    if(is_train){
        data_arg = "--train_data_path";
        labels_arg = "--train_labels_path";
    } else {
        data_arg = "--val_data_path";
        labels_arg = "--val_labels_path";
    }

    int data_pos = find_arg(argv, argc, data_arg);
    int labels_pos = find_arg(argv, argc, labels_arg);

    if(data_pos == -1){
        std::stringstream ss;
        ss<<"can't find parameter "<<data_arg;
        throw std::runtime_error(ss.str());
    }

    if(labels_pos == -1){
        std::stringstream ss;
        ss<<"can't find parameter "<<labels_arg;
        throw std::runtime_error(ss.str());
    }

    data_path = std::string(argv[data_pos+1]);
    labels_path = std::string(argv[labels_pos+1]);
}

void get_layers(char** argv, int argc, std::vector<uint32_t>& layers){
    int pos = find_arg(argv, argc, "--layers=");
    if(pos == -1){
        throw std::runtime_error("can't fine --layers= parameter");
    }
    int offset = 9; // len of --layers=

    std::string param(argv[pos]);
    int next_delim = param.find(',', offset);

    while(offset != std::string::npos){
        layers.push_back(std::stoul(param.substr(offset, next_delim-offset)));

        if(next_delim != std::string::npos)offset = next_delim + 1;
        else offset = next_delim;

        next_delim = param.find(',', offset);
    }
}

void get_activations(char** argv, int argc, std::vector<std::string>& activations){
    int pos = find_arg(argv, argc, "--activations=");
    if(pos == -1){
        throw std::runtime_error("can't fine --activations= parameter");
    }
    int offset = 14; // len of activations or --activations=

    std::string param(argv[pos]);
    int next_delim = param.find(',', offset);

    while(offset != std::string::npos){
        activations.push_back(param.substr(offset, next_delim-offset));

        if(next_delim != std::string::npos)offset = next_delim + 1;
        else offset = next_delim;

        next_delim = param.find(',', offset);
    }
}

void get_num_parameter(char **argv, int argc, const std::string& param_name, int &param) {
    int pos = find_arg(argv, argc, param_name);
    if(pos == -1){
        std::stringstream ss;
        ss<<"can't find parameter "<<param_name;
        throw std::runtime_error(ss.str());
    }
    param = std::stoi(std::string(argv[pos + 1]));
}

void get_num_parameter(char **argv, int argc, const std::string& param_name, float &param) {
    int pos = find_arg(argv, argc, param_name);
    if(pos == -1){
        std::stringstream ss;
        ss<<"can't find parameter "<<param_name;
        throw std::runtime_error(ss.str());
    }
    param = std::stof(std::string(argv[pos + 1]));
}

void read_data(const std::string& data_path, uint32_t data_size, uint32_t data_dim, std::vector<std::vector<float>>& data){
    std::ifstream data_input(data_path);

    if(!data_input.is_open())throw std::runtime_error("can't read data");

    data = std::move(std::vector(data_size, std::vector<float>(data_dim)));

    for(int i = 0;i<data_size;i++){
        for(int j = 0;j<data_dim;j++){
            data_input>>data[i][j];
        }
    }
}

int main(int argc, char** argv) {
    std::vector<uint32_t> layers;
    get_layers(argv, argc, layers);

    std::vector<std::string> activations;
    get_activations(argv, argc, activations);

    int train_dataset_size;
    std::string train_data_path;
    std::string train_labels_path;

    int x_dim;
    int y_dim;

    get_num_parameter(argv, argc, "--x_dim", x_dim);
    get_num_parameter(argv, argc, "--y_dim", y_dim);

    get_num_parameter(argv, argc, "--train_dataset_size", train_dataset_size);

    get_dataset(argv, argc, train_data_path, train_labels_path, true);

    int val_dataset_size;
    std::string val_data_path;
    std::string val_labels_path;

    get_num_parameter(argv, argc, "--val_dataset_size", val_dataset_size);
    get_dataset(argv, argc, val_data_path, val_labels_path);

    int batch_size;
    get_num_parameter(argv, argc, "--batch_size", batch_size);

    int optimization_steps;
    get_num_parameter(argv, argc, "--optimization_steps", optimization_steps);

    float rl;
    get_num_parameter(argv, argc, "--learning_rate", rl);

    MLP mlp = MLP(x_dim, batch_size, layers, activations);

    mlp.forward_initialize();
    std::vector<example> train_dataset(train_dataset_size);


    std::vector<std::vector<float>> train_x;
    std::vector<std::vector<float>> val_x;
    std::vector<std::vector<float>> train_y;
    std::vector<std::vector<float>> val_y;

    read_data(train_data_path, train_dataset_size, x_dim, train_x);
    read_data(train_labels_path, train_dataset_size, y_dim, train_y);

    read_data(val_data_path, val_dataset_size, x_dim, val_x);
    read_data(val_labels_path, val_dataset_size, y_dim, val_y);

    for(int i = 0;i<train_dataset_size;i++){
        train_dataset[i] = std::move(example{.x=train_x[i], .y=train_y[i]});
    }

    std::cout<<"accuracy before training: "<<mlp.evaluate(val_x, val_y)<<std::endl;

    std::unordered_map<std::string, float> optimizer_params;
    optimizer_params["learning_rate"] = rl;

    ClassificationTrainer trainer = ClassificationTrainer(&mlp, train_dataset, optimizer_params);

    std::vector<float> loss_history;

    trainer.train(optimization_steps, loss_history, 100);

    std::cout<<"accuracy after training: "<<mlp.evaluate(val_x, val_y)<<std::endl;

    return 0;
}
