
# Neural Network with One Hidden Layer

## Overview
This project is a Python implementation of a simple neural network with one hidden layer. It demonstrates the fundamental concepts of feed-forward and back-propagation using matrix operations. The primary focus is on understanding how matrix multiplication plays a crucial role in neural network computations.

## Features
- **Sigmoid Activation**: Utilizes the sigmoid function for non-linear activation.
- **Matrix Operations**: Employs matrix multiplication for calculating the feed-forward step.
- **Back-propagation**: Implements the back-propagation algorithm for network training.
- **Error Calculation**: Uses mean absolute error to measure the network's performance.

## Prerequisites
- Python 3.x
- NumPy library

## Installation
To run this project, ensure you have Python installed on your machine. You can download and install Python from [python.org](https://www.python.org/). Additionally, you will need the NumPy library, which can be installed using pip:

```bash
pip install numpy
```

## Usage
To use the neural network, simply clone the repository and run the main script:

```bash
git clone [repository-url]
cd [repository-folder]
python main.py
```

## Example
The current implementation solves the XOR problem. The network takes two binary inputs and predicts a binary output.

## Source
This implementation is inspired by discussions and examples on matrix multiplications in machine learning from [Data Science Stack Exchange](https://datascience.stackexchange.com/questions/75855/what-types-of-matrix-multiplication-are-used-in-machine-learning-when-are-they).
