from math import floor, sqrt
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from collections import Counter
import numpy as np
import re

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


# def X_fit_to_tensor(X_fit):
#     X_fit_tensor = X_fit.reshape(-1, X_fit.shape[0]) 
#     X_fit_tensor = np.expand_dims(X_fit_tensor, axis=-1)
#     X_fit_tensor = np.expand_dims(X_fit_tensor, axis=0)
#     X_fit_tensor = torch.Tensor(X_fit_tensor)
#     return X_fit_tensor

# def eval_classification(pred_func, data):
#     pass

# def eval_regression(pred_fun, data):
#     pass

# def log_classification():
#     pass

def get_max_n_ins(limited_memory, n_features, percision=64):
    # 解析字符串
    # memory_size = re.findall(r'\d+\.*\d*', limited_memory)[0]
    
    memory_size = filter(lambda ch: ch in '0123456789.', limited_memory)
    memory_size = float(''.join(list(memory_size)))
    memory_unit = ''.join(re.findall(r'[A-Za-z]', limited_memory))

    # 转换为bit
    if memory_unit not in ['B', 'KB', 'MB', 'GB']:
        assert False, "Memory unit mast in ['B', 'KB', 'MB', 'GB']"
    elif memory_unit == 'B':
        memory_size = memory_size * 8
    elif memory_unit == 'KB':
        memory_size = memory_size * 1024 * 8
    elif memory_unit == 'MB':
        memory_size = memory_size * 1024 *1024 * 8
    else:
        memory_size = memory_size * 1024 * 1024 * 1024 * 8
    
    # 可以存储的数据量 n_ins * n_features
    max_data_ins = floor(memory_size/percision/(n_features+1))
    # 容许计算km的数据量
    max_km_ins = floor(sqrt(memory_size/percision))

    return min(max_data_ins, max_km_ins) # 二者中最小的

def bit_to_str(num,floor=False):
    # ceil，整数除最高位外均变成0
    num /= 8 # bit to bytes
    if num < 1024:
        num = round(num,2)
        unit = 'B'
    elif num < 1024*1024:
        num = round(num/1024, 2)
        unit = 'KB'
    elif num < 1024*1024*1024:
        num = round(num/1024/1024, 2)
        unit = 'MB'
    else:
        num = round(num/1024/1024/1024, 2)
        unit = 'GB'

    if floor:
        num = int(num)
        len_num = len(str(int(num)))
        num -= (num % 10**(len_num-1))
    
    return str(num) + unit