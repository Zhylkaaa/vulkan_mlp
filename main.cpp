#include <iostream>
#include <MLP.h>
#include <string>

int main() {

    std::vector<int> layers{5, 5};
    std::vector<std::string> activations{"relu", "relu"};
    MLP mlp = MLP(3, 3, layers, activations);

    std::vector<std::vector<float>> batch{{1, 2, 3},
                                          {1, 2, 3},
                                          {3, 2, 1}};

    mlp.forward_initialize();
    mlp.forward(batch);
    mlp.forward(batch);
    mlp.forward(batch);

    return 0;
}
