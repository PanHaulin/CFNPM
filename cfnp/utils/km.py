import torch
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel, sigmoid_kernel

def get_cal_km_func_torch(kernel, gamma, coef0, degree, **kwargs):
        if kernel == 'linear':
            return lambda X_fit, X: torch.mm(X_fit, X.t())
        elif kernel == 'rbf':
            return lambda X_fit, X: torch.exp(-gamma * torch.cdist(X_fit, X, p=2)**2)
        elif kernel == 'poly':
            return lambda X_fit, X: torch.pow(gamma * torch.mm(X_fit, X.t()) + coef0, degree)
        elif kernel == 'sigmoid':
            return lambda X_fit, X: torch.tanh(gamma * torch.mm(X_fit, X.t()) + coef0)
        else:
            assert False, f'want kernel in [linear, rbf, poly, sigmoid], but got {kernel}'
    

# def get_cal_km_func_numpy(kernel, gamma, coef0, degree, **kwargs):
#     if kernel == 'linear':
#         return lambda X_fit, X: np.dot(X_fit, X.T)
#     elif kernel == 'rbf':
#         return lambda X_fit, X: np.exp(-gamma * cdist(X_fit, X.reshape(-1, 1, X.shape[1]), metric='euclidean')).T
#     elif kernel == 'poly':
#         return lambda X_fit, X: np.power(gamma * np.dot(X_fit, X.T) + coef0, degree)
#     elif kernel == 'sigmoid':
#         return lambda X_fit, X: np.tanh(gamma * np.dot(X_fit, X.T) + coef0)
#     else:
#         assert False, f'want kernel in [linear, rbf, poly, sigmoid], but got {kernel}'

def get_cal_km_func_numpy(kernel, gamma, coef0, degree, **kwargs):
    if kernel == 'linear':
        return lambda X_fit, X: linear_kernel(X_fit, X)
    elif kernel == 'rbf':
        return lambda X_fit, X: rbf_kernel(X_fit, X, gamma=gamma)
    elif kernel == 'poly':
        return lambda X_fit, X: polynomial_kernel(X_fit, X, gamma=gamma, coef0=coef0, degree=degree)
    elif kernel == 'sigmoid':
        return lambda X_fit, X: sigmoid_kernel(X_fit, X, gamma=gamma, coef0=gamma)
    else:
        assert False, f'want kernel in [linear, rbf, poly, sigmoid], but got {kernel}'
