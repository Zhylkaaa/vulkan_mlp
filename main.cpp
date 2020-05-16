#include <iostream>
#include <MLP.h>
#include <string>
#include <vulkan_init.h>

int main() {

    std::vector<int> layers{5, 5};
    std::vector<std::string> activations{"relu", "softmax"};
    MLP mlp = MLP(3, 3, layers, activations);

    std::vector<std::vector<float>> batch{{1, 2, 3},
                                          {1, 2, 3},
                                          {3, 2, 1}};

    mlp.forward_initialize();
    mlp.forward(batch);
    mlp.forward(batch);
    mlp.forward(batch);

    VkBuffer labels;
    VkDeviceMemory deviceMemory;

    createBuffer(mlp.get_device(), mlp.get_queue_index(), labels, 3, 5);
    std::vector<VkBuffer*> buffers{&labels};
    std::vector<uint64_t> offsets;

    allocateAndBindBuffers(mlp.get_device(), mlp.get_physicalDevice(), buffers, deviceMemory, offsets);

    char* data = nullptr;
    if(vkMapMemory(mlp.get_device(), deviceMemory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&data)) != VK_SUCCESS){
        throw std::runtime_error("failed to map device memory");
    }

    float* device_labels = reinterpret_cast<float*>(data + offsets[0]);


    std::vector<std::vector<float>> host_labels{{1, 0, 0, 0, 0},
                                   {0, 1, 0, 0, 0},
                                   {0, 0, 1, 0, 0}};

    for(int i = 0;i<host_labels.size();i++){
        for(int j=0;j<host_labels[0].size();j++){
            device_labels[i*5+j] = host_labels[i][j];
        }
    }

    vkUnmapMemory(mlp.get_device(), deviceMemory);

    mlp.backward_initialize(labels);
    mlp.backward();


    return 0;
}
