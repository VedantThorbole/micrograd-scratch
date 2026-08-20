# Micrograd Scratch

A small automatic differentiation engine built from scratch while learning the fundamentals of computational graphs, automatic differentiation, and backpropagation.

This project is my own implementation inspired by Andrej Karpathy's **micrograd** and the *Neural Networks: Zero to Hero* lecture series.

## What I Implemented

The engine contains a `Value` class that supports:

* Computational graph construction
* Automatic differentiation
* Backpropagation
* Topological sorting
* Gradient accumulation
* Operator overloading
* Addition
* Subtraction
* Multiplication
* Power
* Negation
* Division
* Reverse addition
* Reverse multiplication
* Hyperbolic tangent (`tanh`)

## Neural Network Module

Built a small neural network module on top of the automatic differentiation engine.

Implemented:

- Neuron
- Layer
- Multi-Layer Perceptron (MLP)
- Forward propagation

Architecture:

```text
Input
  ↓
Neuron
  ↓
Layer
  ↓
MLP
  ↓
Output
```


Example:

```python
model = MLP(3, [4,4,1])

x = [
    Value(2.0),
    Value(3.0),
    Value(-1.0)
]

output = model(x)
```
Example output:

```text
Output: [Value(-0.29469297479146517)]
```

## Project Structure

```text
micrograd-scratch/
│
├── engine.py
├── nn.py
├── demo.py
├── README.md
├── .gitignore
│
├── tests/
│   ├── __init__.py
│   └── test_engine.py
│
└── examples/
    ├── __init__.py
    └── basic.py
```

## Running the Tests

From the project root:

```bash
python -m unittest discover -s tests -v
```

The test suite covers:

* Addition
* Multiplication
* Subtraction
* Power
* Division
* Tanh
* Chain rule
* Reverse addition
* Reverse multiplication

## Running the Example

From the project root:

```bash
python -m examples.basic
```

## Example Output

```text
output: 0.9999999958776927
d(output)/d(a): 2.4733843639879183e-08
d(output)/d(b): 1.648922909325279e-08
```

## Learning Goal

The goal of this project is not to build a production-ready deep learning framework.

It is a learning implementation to understand what happens underneath higher-level frameworks such as PyTorch:

```text
Operations
    ↓
Computational Graph
    ↓
Local Derivatives
    ↓
Chain Rule
    ↓
Backpropagation
    ↓
Gradients


Neural Network Abstractions

Value
  ↓
Neuron
  ↓
Layer
  ↓
MLP
  ↓
Training
```

## Reference

Inspired by Andrej Karpathy's micrograd and the **Neural Networks: Zero to Hero** lecture series.

This repository is my own implementation created while learning automatic differentiation and backpropagation.

## License / Attribution

This project is inspired by and based on concepts demonstrated in Andrej Karpathy's micrograd project.

If referencing or using code from the original micrograd repository, please follow the original repository's license and attribution requirements.

Original project:

https://github.com/karpathy/micrograd