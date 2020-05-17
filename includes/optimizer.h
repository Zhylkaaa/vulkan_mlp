//
// Created by @Zhylkaaa on 17/05/2020.
//

#ifndef VULKAN_PERCEPTRON_OPTIMIZER_H
#define VULKAN_PERCEPTRON_OPTIMIZER_H
#include <unordered_map>
#include <string>
#include <unordered_set>
#include <vulkan/vulkan.h>

class Optimizer {
protected:
    std::unordered_set<std::string> allowed_params;
    std::unordered_map<std::string, float> optimizer_params;

public:
    void set_parameter(std::string& parameter_name, float new_value);
    void set_parameters(std::unordered_map<std::string, float>& new_params);

    virtual void init(const std::vector<std::pair<VkBuffer, VkBuffer>>& trainable_parameters) = 0;
    virtual void optimize() = 0;
};
#endif //VULKAN_PERCEPTRON_OPTIMIZER_H
