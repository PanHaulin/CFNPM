import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

def save_grad(name, grads):
    def hook(grad):
        grads[name] = grad
    return hook

class CeilNoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.ceil()
    @staticmethod
    def backward(ctx, g):
        return g

class SumOfKlLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, mean1, std1, mean2, std2):
        loss = 0
        for i in range(mean1.size(0)):
            p = Normal(mean1[i][0], std1[i][0])
            q = Normal(mean2[i][0], std2[i][0])
            loss += kl_divergence(p, q)
        return loss