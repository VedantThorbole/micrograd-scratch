from engine import Value
from nn import MLP

# Create a neural network
model = MLP(3, [4, 4, 1])

# Sample input
x = [
    Value(2.0),
    Value(3.0),
    Value(-1.0)
]

# Forward pass
out = model(x)

print("Output:", out)