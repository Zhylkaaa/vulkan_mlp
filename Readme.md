# Vulkan MLP

## Overview 
Simple implementation of Multi-layer perceptron using Vulkan Compute API.

## Usage
For now example of usage can be found in `main.cpp`.<br>
To use this you should compile shaders following name convention:
```$bash
#in shaders folder
$ glslc -o d_softmax.comp.spv d_softmax.comp 
$ glslc -o dense.comp.spv dense.comp 
```

To compile and run use:
```bash
$ chmod u+x compile_shaders.sh
$ ./compile_shaders.sh
$ mkdir build
$ cd build
$ cmake ..
$ make
$ ./vulkan_perceptron
```

## Quick reference

to define perceptron you need vector with output sizes of each layer (input size will be inferred from previous layer)
and vector of strings containing names of activations for each layer.
also you need to specify input size and batch size (**Note:** that for sake of simplicity you can use forward only with matrix of size `batch_size x input_size`)

Activation list:
- relu - Rectified linear unit 
- id - identity (not a layer actually, it's just omitted)
- softmax - softmax function, not actually an activation function, but it's hear for sake of simplicity

You can refer to `line 10,11,12 of main.cpp` for usage example.

## TODOs
- Switch to using Tensor instead of VkBuffer
- Implement trainers for different tasks
- BIG code refactor
- Implement different activations
