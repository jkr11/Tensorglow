import torch
from tensor import Tensor
import numpy as np

xi = np.random.randn(1,3).astype(np.float32)
Wi = np.random.randn(3,3).astype(np.float32)
mi = np.random.randn(1,3).astype(np.float32)

def test_tens():
    x = Tensor(xi)
    W = Tensor(Wi)
    m = Tensor(mi)
    out = x.dot(W)
    outr = out.relu()
    outsm = outr.logsoftmax()
    outm = outsm.mul(m)
    outs = outm.sum()
    return outs.data, x.grad, W.grad

def test_torch():
    x = torch.tensor(xi, requires_grad=True)
    W = torch.tensor(Wi, requires_grad=True)
    m = torch.Tensor(mi)
    out = x @ W
    outs = out.sum()
    outs.backward()
    return outs.detach().numpy(), x.grad, W.grad

for i,j in zip(test_tens(), test_torch()):
    print(i)