My AI Learning Journey: Neural Networks from Scratch in NumPy

https://neuralnetworknromscratchinnumpy.netlify.app/

This repository is a log of my intensive journey to build a complete deep learning framework from the ground up, using only NumPy.

This repo contains the evolution of that work, from a simple 2-layer network to a complete, Numba-accelerated Convolutional Neural Network (CNN) that can train on the MNIST dataset.

🧠 Core Concepts I Implemented from Scratch

Dense (Fully-Connected) Layers: The core y = mx + c logic (np.dot(inputs, weights) + biases).

Activation Functions: Sigmoid, Tanh, ReLU, and LeakyReLU (and their derivatives).

Full Backpropagation: A complete, modular backward() method for every single layer.

Loss Function: A stable Softmax + Categorical Cross-Entropy loss class.

Modern Optimizer: A full implementation of the Adam optimizer, including momentum (m), RMSprop (v), and bias correction (beta1, beta2).

Regularization:

L2 Regularization ("Weight Decay"): Integrated into the loss and gradient calculations.

Dropout: A full dropout layer with a training_mode switch.

Convolutional Layers:

Conv2D (forward and backward pass).

MaxPool2D (forward and backward pass).

Performance Optimization: Accelerated the slow Python loops in CNNs by 100x+ using Numba (@jit, prange).

🚀 The Projects

This repository contains the code for two main projects that evolved over time.

Project 1: Deep Neural Network (DNN) for Student Pass/Fail

This was the first model, a multi-layer perceptron (MLP) built to learn the fundamentals.

Goal: Predict if a student would "Pass" or "Fail" based on Score and Study_Hours.

Architecture: A 2-16-16-2 network (2 inputs, 2 hidden layers of 16 neurons, 2 outputs).

Key Learnings:

Implemented the full Forward Pass and Backward Pass manually.

Experimented with Sigmoid and Tanh and discovered the Vanishing Gradient Problem (training failed).

Solved this by implementing LeakyReLU, which allowed the network to train successfully.

Implemented Adam, L2, and Dropout to create a modern, well-regulated training process.

Project 2: Convolutional Neural Network (CNN) for MNIST

This is the "final boss" project, combining all concepts to tackle a real-world computer vision problem.

Goal: Classify handwritten digits (0-9) from the MNIST dataset.

Data Pipeline: Loads the mnist_train.csv file, normalizes pixel data (/ 255.0), and reshuffles in batches.

New Layers:

Conv2D: A class to perform convolution (feature extraction).

MaxPool2D: A class to downsample the feature maps.

Flatten: A layer to connect the 2D conv layers to the 1D dense layers.

Performance: The pure Python for loops in Conv2D were too slow (estimated 8-20 hours). I solved this by implementing Numba (@jit, prange) to compile the loops, accelerating the training to a usable speed.

✍️ My Learning Process (The "Why")

This repository isn't just a collection of scripts; it's proof of my learning philosophy. While many tutorials (the "Race Car Driver" path) teach you to just use a library like Keras, I chose the "Engine Mechanic" path.

To truly understand how the engine worked, I derived the math for backpropagation and the Adam optimizer by hand over 20+ pages of handwritten notes.

By building the engine from scratch, I now have a deep, fundamental understanding of why it works, which will make me a better engineer when I move on to using high-level frameworks.
