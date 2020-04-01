import torch
from torch.nn import Sigmoid

A = torch.tensor([1,2,3], dtype=torch.float64)
f = Sigmoid()
print(A)
print(f(A))