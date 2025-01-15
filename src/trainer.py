from engine import *


xs = [
    [random.randint(-10, 20) for _ in range(4)],
    [random.randint(10, 30) for _ in range(4)],
    [random.randint(-15, 15) for _ in range(4)],
    [random.randint(5, 10) for _ in range(4)],
]

ys = [
    -1,
    1,
    -1,
    1
]

m = MLP(4, [4, 4, 1])
lr = 0.01
for i in range(30):
    out = [m(x) for x in xs]
    loss = sum([(p - y)**2 for p, y in zip(out, ys)], Value(0))
    for p in m.parameters():
        p.grad = 0

    loss.backward()
    for p in m.parameters():
        p.data = p.data - lr * p.grad
print(loss.data)

for x,y in zip(xs, ys):
    print(m(x), y)