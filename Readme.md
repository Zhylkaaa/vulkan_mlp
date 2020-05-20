# Vulkan MLP

## Overview 
Simple implementation of Multi-layer perceptron using Vulkan Compute API.

## Usage
For now example of usage can be found in `main.cpp`.<br>

To compile and run use:
```bash
$ chmod u+x compile_shaders.sh
$ ./compile_shaders.sh
$ mkdir build
$ cd build
$ cmake ..
$ make
```

To check if it works you can run example:

```bash
$ ./vulkan_perceptron \
--train_dataset_size 20000 \
--val_dataset_size 10000 \
--train_data_path ../train_MNIST_images.txt \
--train_labels_path ../train_MNIST_labels.txt \
--val_data_path ../val_MNIST_images.txt \
--val_labels_path ../val_MNIST_labels.txt \
--batch_size 32 \
--x_dim 784 \
--y_dim 10 \
--learning_rate 0.3 \
--optimization_steps 1000 \
--layers=512,10 \
--activations=relu,softmax
```

## Quick reference

to define perceptron you need vector with output sizes of each layer (input size will be inferred from previous layer)
and vector of strings containing names of activations for each layer.
also you need to specify input size and batch size (**Note:** that for sake of simplicity you can use forward only with matrix of size `batch_size x input_size`)

Activation list:
- relu - Rectified linear unit 
- id - identity (not a layer actually, it's just omitted)
- softmax - softmax function, not actually an activation function, but it's hear for sake of simplicity

## TODOs
- Switch to using Tensor instead of VkBuffer
- Implement trainers for different tasks
- BIG code refactor
- Implement different activations
