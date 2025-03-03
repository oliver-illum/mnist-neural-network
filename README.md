# MNIST Neural Network
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

![Diagram](assets/mnist-readme.png)

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Neural Network Structure](#neural-network-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results](#results)

## Overview

This repository is an experimental project that demonstrates how a feedforward (fully connected) neural network can be implemented using different programming languages with the MNIST dataset. Currently, the project includes implementations in Rust and Python. The focus is on exploring how neural networks can be designed and constructed in various programming environments. Although the current implementation uses a feedforward approach, the repository is structured with future extensions in mind, making it possible to incorporate multiple neural network algorithms and architectures as the project evolves.

While the MNIST dataset is sometimes viewed as overused, its simplicity and well-known structure make it an ideal starting point for understanding the core mechanics of neural network design and experimentation.

## Project Structure

The repository is organized into several key components:

### data Folder
- **Description:** Contains the MNIST dataset files in IDX format. Both implementations load the data from this common folder, ensuring consistency in training and evaluation.

### download_data.sh
- **Description:** *(Optional)* A shell script that automates downloading and extracting the MNIST dataset files. This script ensures that both the Rust and Python implementations work with the correct dataset.

### python_feedforward
- **Description:** Contains the Python implementation of the neural network. The core code is located in the `src` subfolder, which includes modules for activations, dataset handling, layers, loss functions, network assembly, and utility functions.

### rust_feedforward
- **Description:** Contains the Rust implementation of the neural network. Similar to the Python side, the Rust code in the `src` folder is organized into modules for activations, dataset handling, layers, loss functions, network assembly, and utilities.

## Neural Network Structure

For now, this repository features a single feedforward (fully connected) neural network. The structure of the network is consistent across both implementations. The following files (located in each language's `src` directory) describe the main components:

### main (main.rs / main.py)
- **Description:** The entry point of the application. It initializes the network, loads the MNIST dataset, runs the training loop, and outputs performance metrics such as loss and accuracy.

### activations (activations.rs / activations.py)
- **Description:** Contains the activation functions (e.g., ReLU, Sigmoid) that introduce non-linearity into the network. These functions are applied after the weighted sums in each layer to enable learning of complex patterns.

### dataset (dataset.rs / dataset.py)
- **Description:** Handles loading and preprocessing of the MNIST dataset. This module reads the IDX files from the shared `data` folder, normalizes the image data, and partitions it into training and test sets.

### layer (layer.rs / layer.py)
- **Description:** Defines the structure and operations of a single neural network layer. This includes managing both the forward propagation (computing outputs) and the backward propagation (calculating gradients for weight updates).

### network (network.rs / network.py)
- **Description:** Assembles the individual layers into a complete feedforward neural network. This module outlines the overall architecture and coordinates the data flow through the network.

### loss (loss.rs / loss.py)
- **Description:** Implements the loss functions (such as cross-entropy) used to measure the error between the predicted outputs and the actual labels. It also calculates gradients used in the backpropagation process.

### utils (utils.rs / utils.py)
- **Description:** Provides utility functions that support common tasks, such as parameter initialization, logging, and performance tracking. These helper functions keep the main code clean and modular.

## Installation & Setup

## Results
