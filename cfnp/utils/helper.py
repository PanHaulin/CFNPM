import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from collections import Counter
import numpy as np

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

def multi_to_binary(y: np.ndarray) -> np.ndarray:
    sorted_kv = sorted(Counter(y).items(), reverse=True)
    label = sorted_kv[0][0]
    num_true = sorted_kv[0][1]
    num_false = 0
    labels_true = []
    labels_false = []
    labels_true.append(label)
    for i in range(1, len(sorted_kv)):
        label = sorted_kv[i][0]
        num = sorted_kv[i][1]
        if num_true > num_false:
            labels_false.append(label)
            num_false += num
        else:
            labels_true.append(label)
            num_true += num
    return labels_true, labels_false


def X_fit_to_tensor(X_fit):
    X_fit_tensor = X_fit.reshape(-1, X_fit.shape[0]) 
    X_fit_tensor = np.expand_dims(X_fit_tensor, axis=-1)
    X_fit_tensor = np.expand_dims(X_fit_tensor, axis=0)
    X_fit_tensor = torch.Tensor(X_fit_tensor)
    return X_fit_tensor

def eval_classification(pred_func, data):
    pass

def eval_regression(pred_fun, data):
    pass

def log_classification():
    pass
