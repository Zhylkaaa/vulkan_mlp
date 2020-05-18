//
// Created by @Zhylkaaa on 17/05/2020.
//

#include <optimizer.h>

void Optimizer::set_parameter(std::string &parameter_name, float new_value) {
    if(allowed_params.find(parameter_name) == allowed_params.end())
        throw std::invalid_argument("unknown parameter");
    optimizer_params[parameter_name] = new_value;
}

void Optimizer::set_parameters(const std::unordered_map<std::string, float> &new_params) {
    for(auto param : new_params){
        if(allowed_params.find(param.first) == allowed_params.end())
            throw std::invalid_argument("unknown parameter");
        optimizer_params.insert(param);
    }
}
