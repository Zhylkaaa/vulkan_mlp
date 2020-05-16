//
// Created by Dima Zhylko on 12/05/2020.
//

#include <vulkan_init.h>
#include <vulkan/vulkan.h>

#include <vector>
#include <iostream>
#include <fstream>

const std::vector<const char *> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
        void *pUserData) {

    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

bool checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);

    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char *layerName : validationLayers) {
        bool found = false;
        for (const auto &layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                found = true;
                break;
            }
        }

        if (!found) {
            return false;
        }
    }
    return true;
}

std::vector<const char *> getRequiredExtensions() {

    std::vector<const char *> extensions;

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo) {
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
}

void createInstance(VkInstance& instance) {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vector Add";
    appInfo.engineVersion = VK_API_VERSION_1_0;
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_API_VERSION_1_0;
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *) &debugCreateInfo;
    } else {
        createInfo.enabledLayerCount = 0;

        createInfo.pNext = nullptr;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
    }
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
                                      const VkAllocationCallbacks *pAllocator,
                                      VkDebugUtilsMessengerEXT *pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance,
                                                                           "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

void setupDebugMessenger(const VkInstance& instance, VkDebugUtilsMessengerEXT& debugMessenger) {
    if (!enableValidationLayers) return;
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    populateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
        throw std::runtime_error("failed to set up debug messenger!");
    }
}

bool isDeviceSuitable(VkPhysicalDevice device, uint32_t& queueFamilyIndex){
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for(const auto& queueFamily : queueFamilies){
        if(queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT){
            queueFamilyIndex = i;
            return true;
        }
        i++;
    }

    return false;
}

void pickPhysicalDevice(const VkInstance& instance, VkPhysicalDevice& physicalDevice, uint32_t& queueFamilyIndex) {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    physicalDevice = VK_NULL_HANDLE;

    for (const auto &device : devices) {
        if (isDeviceSuitable(device, queueFamilyIndex)) {
            physicalDevice = device;
            break;
        }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }

    VkPhysicalDeviceProperties gpuProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &gpuProperties);

    std::cout << "Using device: " << gpuProperties.deviceName << '\n';
}

void createLogicalDeviceAndQueue(const VkInstance& instance, const VkPhysicalDevice& physicalDevice,
        uint32_t& queueFamilyIndex, VkDevice& device, VkQueue& queue){

    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

    if (enableValidationLayers) {
        deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        deviceCreateInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        deviceCreateInfo.enabledLayerCount = 0;
    }

    if(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS){
        throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
}

void setup_vulkan(VkInstance& instance, VkDebugUtilsMessengerEXT& debugMessenger, VkPhysicalDevice& physicalDevice,
                  uint32_t& queueFamilyIndex, VkDevice& device, VkQueue& queue){

    createInstance(instance);
    setupDebugMessenger(instance, debugMessenger);
    pickPhysicalDevice(instance, physicalDevice, queueFamilyIndex);
    createLogicalDeviceAndQueue(instance, physicalDevice, queueFamilyIndex, device, queue);
}

void createBuffer(const VkDevice& device, uint32_t queueFamilyIndex, VkBuffer& buffer,
        uint32_t n, uint32_t m, uint64_t elem_size){
    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = n*m*elem_size;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferCreateInfo.queueFamilyIndexCount = 1;
    bufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;

    if(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer) != VK_SUCCESS){
        throw std::runtime_error("failed to create buffer");
    }
}

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void allocateAndBindBuffers(const VkDevice& device, const VkPhysicalDevice& physicalDevice, std::vector<VkBuffer*>& buffers,
        VkDeviceMemory& memory, std::vector<uint64_t>& offsets){

    VkDeviceSize requiredMemorySize = 0;
    uint32_t typeFilter = 0;

    for(VkBuffer* buff : buffers){
        VkMemoryRequirements bufferMemoryRequirements;

        vkGetBufferMemoryRequirements(device, *buff, &bufferMemoryRequirements);
        requiredMemorySize += bufferMemoryRequirements.size;

        if(bufferMemoryRequirements.size % bufferMemoryRequirements.alignment != 0){
            requiredMemorySize += bufferMemoryRequirements.alignment - bufferMemoryRequirements.size % bufferMemoryRequirements.alignment;
        }
        typeFilter |= bufferMemoryRequirements.memoryTypeBits;
    }

    uint32_t memoryTypeIndex = findMemoryType(physicalDevice, typeFilter, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = requiredMemorySize;
    allocateInfo.memoryTypeIndex = memoryTypeIndex;

    if (vkAllocateMemory(device, &allocateInfo, nullptr, &memory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory");
    }

    VkDeviceSize offset = 0;

    for(VkBuffer* buff : buffers){
        offsets.push_back(static_cast<uint32_t>(offset));

        VkMemoryRequirements bufferMemoryRequirements;
        vkGetBufferMemoryRequirements(device, *buff, &bufferMemoryRequirements);

        if(vkBindBufferMemory(device, *buff, memory, offset) != VK_SUCCESS){
            throw std::runtime_error("failed to bind buffer memory");
        }

        offset += bufferMemoryRequirements.size;
        if(bufferMemoryRequirements.size % bufferMemoryRequirements.alignment != 0){
            offset += bufferMemoryRequirements.alignment - bufferMemoryRequirements.size % bufferMemoryRequirements.alignment;
        }
    }
}

void createPipelineLayout(const VkDevice& device, uint32_t bindingsCount, VkDescriptorSetLayout& setLayout,
        VkPipelineLayout& pipelineLayout, uint32_t push_constant_size){
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

    for(uint32_t i = 0;i<bindingsCount;i++){
        VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = i;
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBinding.descriptorCount = 1;
        layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        layoutBindings.push_back(layoutBinding);
    }

    VkDescriptorSetLayoutCreateInfo setLayoutCreateInfo{};
    setLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setLayoutCreateInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
    setLayoutCreateInfo.pBindings = layoutBindings.data();

    if(vkCreateDescriptorSetLayout(device, &setLayoutCreateInfo, nullptr, &setLayout) != VK_SUCCESS){
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &setLayout;

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.size = push_constant_size;
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;

    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

    if(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout)){
        throw std::runtime_error("failed to create pipeline layout");
    }
}

std::vector<char> readFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

void createComputePipeline(const VkDevice& device, const std::string& shaderName, const VkPipelineLayout& pipelineLayout,
        VkPipeline& pipeline, const std::string& entry_point){
    VkShaderModuleCreateInfo shaderModuleCreateInfo{};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

    auto shaderCode = readFile(shaderName);
    shaderModuleCreateInfo.pCode = reinterpret_cast<uint32_t*>(shaderCode.data());
    shaderModuleCreateInfo.codeSize = shaderCode.size();

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if(vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule) != VK_SUCCESS){
        throw std::runtime_error("failed to create shader module");
    }

    VkComputePipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineCreateInfo.stage.module = shaderModule;

    pipelineCreateInfo.stage.pName = entry_point.c_str();
    pipelineCreateInfo.layout = pipelineLayout;

    if(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline) != VK_SUCCESS){
        throw std::runtime_error("failed to create compute pipeline");
    }
    vkDestroyShaderModule(device, shaderModule, nullptr);
}

void allocateDescriptorSet(const VkDevice& device, std::vector<VkBuffer*>& buffers,
        VkDescriptorPool& descriptorPool, const VkDescriptorSetLayout &setLayout, VkDescriptorSet& descriptorSet){
    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets = 1;

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = static_cast<uint32_t>(buffers.size());

    descriptorPoolCreateInfo.poolSizeCount = 1;
    descriptorPoolCreateInfo.pPoolSizes = &poolSize;
    if(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS){
        throw std::runtime_error("failed to create descriptor pool");
    }

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = &setLayout;

    if(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet) != VK_SUCCESS){
        throw std::runtime_error("failed to allocate descriptor sets");
    }


    std::vector<VkWriteDescriptorSet> descriptorSetWrites(buffers.size());
    std::vector<VkDescriptorBufferInfo> bufferInfos(buffers.size());

    uint32_t i = 0;
    for(VkBuffer* buff : buffers){
        VkWriteDescriptorSet writeDescriptorSet{};
        writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.dstSet = descriptorSet;
        writeDescriptorSet.dstBinding = i;
        writeDescriptorSet.dstArrayElement = 0;
        writeDescriptorSet.descriptorCount = 1;
        writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

        VkDescriptorBufferInfo buffInfo{};
        buffInfo.buffer = *buff;
        buffInfo.offset = 0;
        buffInfo.range = VK_WHOLE_SIZE;
        bufferInfos[i] = buffInfo;

        writeDescriptorSet.pBufferInfo = &bufferInfos[i];
        descriptorSetWrites[i] = writeDescriptorSet;
        i++;
    }

    vkUpdateDescriptorSets(device, descriptorSetWrites.size(), descriptorSetWrites.data(), 0, nullptr);
}

void createCommandPoolAndBuffer(const VkDevice& device, uint32_t queueFamilyIndex,
        VkCommandPool& commandPool, VkCommandBuffer& commandBuffer, VkCommandPoolCreateFlags flags){
    VkCommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = flags;
    commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;

    if(vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool) != VK_SUCCESS){
        throw std::runtime_error("failed to create command pool");
    }

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;

    if(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer) != VK_SUCCESS){
        throw std::runtime_error("failed to allocate command buffer");
    }
}

void recordComputePipeline(VkCommandBuffer& commandBuffer, const VkPipelineLayout& pipelineLayout,
        uint32_t push_constant_size, void* push_constant_vals, const VkPipeline& pipeline,
        VkDescriptorSet& descriptorSet, uint32_t x_group, uint32_t y_group, uint32_t z_group,
        VkCommandBufferUsageFlags flags){
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.flags = flags;
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if(vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS){
        throw std::runtime_error("failed to begin command buffer");
    }

    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_size, push_constant_vals);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdDispatch(commandBuffer, x_group, y_group, z_group);

    if(vkEndCommandBuffer(commandBuffer) != VK_SUCCESS){
        throw std::runtime_error("failed to end command buffer");
    }
}

void submitTask(VkQueue& queue, VkCommandBuffer* pCommandBuffer, bool wait_for_queue){
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = pCommandBuffer;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    // TODO: use fence?
    if(wait_for_queue)vkQueueWaitIdle(queue);
}