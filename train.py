def loss(model):

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

    total_loss = 0

    for x, y in zip(xs, ys):
        pred = model(x)
        sample_loss = (pred[0] - y) ** 2
        total_loss += sample_loss
    return total_loss / len(xs)



def training(model, learning_rate, epochs=100):

    for epoch in range(epochs):
        for p in model.parameters():
            p.grad = 0

        l = loss(model)       
        l.backward()
        
        for p in model.parameters():
            p.data += -learning_rate * p.grad

        if (epoch + 1) % 10 == 0:
            print(epoch + 1, l.data)