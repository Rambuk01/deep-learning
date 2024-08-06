import torch

x = torch.rand(2, 2)
print(x)
y = torch.rand(2, 2)
print(y)
b = torch.matmul(x, y)

print(b)