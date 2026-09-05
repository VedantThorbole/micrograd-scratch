# Micrograd Scratch

A small automatic differentiation engine and neural network framework built completely from scratch to understand the fundamentals behind deep learning.

This project is my own implementation inspired by Andrej Karpathy's **micrograd** and the **Neural Networks: Zero to Hero** lecture series.

The goal of this project is to understand what happens internally inside frameworks like PyTorch by implementing:

```
Operations
    ↓
Computational Graph
    ↓
Automatic Differentiation
    ↓
Chain Rule
    ↓
Backpropagation
    ↓
Gradient Descent
    ↓
Neural Network Training
```

---

# Features Implemented

## Automatic Differentiation Engine

A custom `Value` class implementing:

- Computational graph construction
- Automatic differentiation
- Backpropagation
- Topological sorting
- Gradient accumulation
- Operator overloading

Supported mathematical operations:

- Addition
- Subtraction
- Multiplication
- Division
- Power
- Negation
- Reverse addition
- Reverse multiplication
- Hyperbolic tangent (`tanh`)

---

# Neural Network Module

Built a simple neural network library on top of the autograd engine.

Implemented:

- Neuron
- Layer
- Multi-Layer Perceptron (MLP)
- Forward propagation
- Parameter collection

Architecture:

```
Input
  |
  ↓
Neuron
  |
  ↓
Layer
  |
  ↓
MLP
  |
  ↓
Prediction
```

---

# Model Architecture

Example:

```python
model = MLP(3, [4,4,1])
```

Network:

```
Input Layer
(3 inputs)

        ↓

Hidden Layer
(4 neurons)

        ↓

Hidden Layer
(4 neurons)

        ↓

Output Layer
(1 neuron)
```

Total trainable parameters:

```
41 parameters
```

---

# Training Pipeline

The model is trained using gradient descent.

Training flow:

```
Input Data

    ↓

Forward Pass

    ↓

Prediction

    ↓

Mean Squared Error Loss

    ↓

Backward Pass

    ↓

Gradient Calculation

    ↓

Parameter Update

    ↓

Repeat
```

Gradient update:

```
parameter =
parameter - learning_rate * gradient
```

---

# Example Training

The model learns from a small dataset:

```python
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]


ys = [
    1.0,
    -1.0,
    -1.0,
    1.0
]
```

Training:

```python
training(model, 0.01)
```

Example output:

```
10 0.8231
20 0.2145
30 0.0632
40 0.0189
50 0.0061
```

The decreasing loss shows that the neural network is learning by updating its parameters through gradients.

---

# Project Structure

```
micrograd-scratch/

│
├── engine.py
│   └── Automatic differentiation engine
│
├── nn.py
│   └── Neural network components
│       ├── Neuron
│       ├── Layer
│       └── MLP
│
├── train.py
│   └── Loss function and training loop
│
├── demo.py
│   └── Model training example
│
├── tests/
│   ├── __init__.py
│   └── test_engine.py
│
└── README.md
```

---

# Running the Project

Clone the repository:

```bash
git clone https://github.com/VedantThorbole/micrograd-scratch.git
```

Go inside the project:

```bash
cd micrograd-scratch
```

Run example:

```bash
python demo.py
```

---

# Example Usage

Create a neural network:

```python
from nn import MLP


model = MLP(3,[4,4,1])
```

Forward pass:

```python
output = model([
    Value(2.0),
    Value(3.0),
    Value(-1.0)
])
```

Output:

```
[Value(...)]
```

---

# Learning Journey

This project was built to understand the foundations of neural networks:

```
Value
 ↓
Computational Graph
 ↓
Neuron
 ↓
Layer
 ↓
MLP
 ↓
Loss Function
 ↓
Backpropagation
 ↓
Gradient Descent
 ↓
Training
```

After implementing these concepts from scratch, higher-level frameworks like PyTorch become easier to understand because the underlying mechanics are no longer a black box.

---

# Future Improvements

Possible future additions:

- Better optimizer support
- Mini-batch training
- More activation functions
- Softmax and classification loss
- Dataset abstraction
- Model saving/loading

---

# Reference

Inspired by:

Andrej Karpathy - micrograd

https://github.com/karpathy/micrograd

Neural Networks: Zero to Hero

---

# Attribution

This repository is my own implementation created for learning automatic differentiation, neural networks, and backpropagation.

The project is inspired by educational concepts demonstrated in Andrej Karpathy's work and the original micrograd repository.


## License / Attribution

This project is inspired by and based on concepts demonstrated in Andrej Karpathy's micrograd project.

If referencing or using code from the original micrograd repository, please follow the original repository's license and attribution requirements.

Original project:

https://github.com/karpathy/micrograd