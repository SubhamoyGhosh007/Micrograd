# Micrograd from Scratch

A minimal implementation of an automatic differentiation engine and a multi-layer perceptron (MLP) neural network built entirely from scratch in Python, inspired by Andrej Karpathy's Micrograd project.

This project demonstrates the core concepts behind deep learning frameworks such as PyTorch by implementing:

* Reverse-mode automatic differentiation (backpropagation)
* Computational graph construction
* Gradient computation
* Neurons, layers, and multi-layer perceptrons
* Gradient descent optimization
* Graph visualization using Graphviz

---

## Features

### Automatic Differentiation Engine

The `Value` class forms the foundation of the project.

Supported operations:

* Addition (`+`)
* Subtraction (`-`)
* Multiplication (`*`)
* Division (`/`)
* Power (`**`)
* Negation (`-x`)
* Exponential (`exp`)
* Hyperbolic tangent (`tanh`)

Each operation:

1. Creates a node in the computational graph.
2. Stores references to parent nodes.
3. Defines a local backward function.
4. Enables automatic gradient computation through reverse-mode differentiation.

---

## Neural Network Components

### Neuron

Each neuron contains:

* Trainable weights
* Bias term
* Tanh activation function

Forward computation:

```text
output = tanh(w₁x₁ + w₂x₂ + ... + b)
```

### Layer

A collection of neurons operating in parallel.

### MLP (Multi-Layer Perceptron)

Supports arbitrary network architectures.

Example:

```python
n = MLP(3, [4, 4, 1])
```

Creates:

```text
Input (3)
   ↓
Hidden Layer (4)
   ↓
Hidden Layer (4)
   ↓
Output Layer (1)
```

---

## Project Structure

```text
.
├── micrograd.py
├── README.md
└── graph.svg (generated visualization)
```

Main classes:

```python
Value
Neuron
Layer
MLP
```

Utility functions:

```python
trace()
draw_dot()
```

---

## Computational Graph Visualization

The project uses Graphviz to visualize the computational graph.

Example:

```python
dot = draw_dot(loss)
dot.render("graph", view=True)
```

Visualization includes:

* Node values
* Gradients
* Operations
* Dependency relationships

---

## Training Example

Training dataset:

```python
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]
```

Network:

```python
n = MLP(3, [4, 4, 1])
```

Loss function:

```python
loss = Σ(y_pred - y_true)²
```

Optimization:

```python
p.data += -0.05 * p.grad
```

Training runs for:

```python
epoch = 100
```

---

## Backpropagation

Backward propagation is implemented through:

```python
loss.backward()
```

The algorithm:

1. Builds a topological ordering of the computational graph.
2. Initializes the output gradient to 1.
3. Traverses the graph in reverse order.
4. Applies the chain rule to compute gradients.

---

## Installation

Clone the repository:

```bash
git clone git@github.com:SubhamoyGhosh007/Micrograd.git
cd Micrograd
```

Install dependencies:

```bash
pip install graphviz
```

Install Graphviz system package:

### Ubuntu

```bash
sudo apt install graphviz
```

### macOS

```bash
brew install graphviz
```

### Windows

Download and install Graphviz:

https://graphviz.org/download/

---

## Running the Project

```bash
python micrograd.py
```

You will see:

* Predictions for each epoch
* Loss values
* Updated network parameters through gradient descent

---

## Learning Objectives

This project helps understand:

* How neural networks work internally
* Computational graphs
* Automatic differentiation
* Backpropagation
* Gradient descent optimization
* Deep learning fundamentals without external ML libraries

---

## Inspiration

Inspired by Andrej Karpathy's Micrograd project and his Neural Networks: Zero to Hero series.

The goal of this implementation is educational: understanding what happens behind modern deep learning frameworks rather than relying on high-level abstractions.

---

## Future Improvements

* ReLU activation
* Sigmoid activation
* Cross-entropy loss
* Batch training
* SGD / Adam optimizers
* Model serialization
* Mini-batch learning
* Better graph visualization
* Dataset loading utilities

---

## License

MIT License
