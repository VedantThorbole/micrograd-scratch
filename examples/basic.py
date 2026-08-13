from engine import Value

a = Value(2)
b = Value(3)

c = a * b
d = c + 4
e = d.tanh()

e.backward()

print("output:", e.data)
print("d(output)/d(a):", a.grad)
print("d(output)/d(b):", b.grad)
