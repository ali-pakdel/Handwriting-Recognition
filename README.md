# Handwritten Image Recognition using Feed-Forward Neural Networks

This project comprises two phases: Phase 1 involves the implementation of a feed-forward neural network **from scratch** to recognize handwritten numbers, while Phase 2 focuses on implementing a similar network using `Keras` and `TensorFlow` to recognize handwritten alphabets.

## Phase 1: Handwritten Numbers Recognition

In this phase, we implement a feed-forward neural network from scratch using the `Numpy` library to classify images of handwritten numbers. Each image is first flattened as a vector and then given as input to the network. The network adjusts the weights of connections between its layers, making nonlinear combinations, to minimize the error and accurately predict the corresponding handwritten number.

### Objectives

In Phase 1, our primary objectives are as follows:

- Data visualization and preprocessing, including min-max normalization and one-hot encoding of labels.
- Implementation of various activation functions, including `Identical`, `Relu`, `LeakyRelu`, `Sigmoid`, `Softmax`, and `Tanh`.
- Implementation of the `CrossEntropy` loss function.
- Implementation of `Layer` and `FeedForwardNN` components.
- Testing data classification and evaluating the impact of network weighting methods, learning rate, activation function, and batch size.

## Phase 2: Handwritten Alphabets Recognition

Phase 2 aims to recognize handwritten alphabets using the `Keras` and `TensorFlow` libraries. We build a neural network model to accurately classify the letters.

### Objectives

In Phase 2, our primary objectives are as follows:

- Preprocessing tasks, including image grayscaling and one-hot encoding of labels.
- Implementation of a neural network using the `Keras` and `TensorFlow` libraries.
- Data classification and evaluation of the impacts of optimizer, number of epochs, loss function, and regularization.
- Dimensionality reduction for data visualization.
