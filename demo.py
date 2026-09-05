from engine import Value
from nn import MLP
from train import training


model = MLP(3, [4, 4, 1])


print("Parameters:", len(model.parameters()))


training(model, 0.01, 100)



print("\nPredictions:")


xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]


for x in xs:
    print(model(x))