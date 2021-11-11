import torch
import numpy as np
from scipy.spatial.distance import cdist

def get_cal_km_func_torch(**params):
        if params['kernel'] == 'linear':
            return lambda X_fit, X: torch.mm(X_fit, X.t())
        elif params['kernel'] == 'rbf':
            return lambda X_fit, X: torch.exp(-params['gamma'] * torch.cdist(X_fit, X, p=2)**2)
        elif params['kernel'] == 'poly':
            return lambda X_fit, X: torch.pow(params['gamma'] * torch.mm(X_fit, X.t()) + params['coef0'], params['degree'])
        elif params['kernel'] == 'sigmoid':
            return lambda X_fit, X: torch.tanh(params['gamma'] * torch.mm(X_fit, X.t()) + params['coef0'])
        else:
            assert False, f'want kernel in [linear, rbf, poly, sigmoid], but got {params["kernel"]}'


def get_cal_km_func_numpy(**params):
    if params['kernel'] == 'linear':
        return lambda X_fit, X: np.dot(X_fit, X.T)
    elif params['kernel'] == 'rbf':
        return lambda X_fit, X: np.exp(-params['gamma'] * cdist(X_fit, X.reshape(-1, 1, X.shape[1]), metric='euclidean')).T
    elif params['kernel'] == 'poly':
        return lambda X_fit, X: np.power(params['gamma'] * np.dot(X_fit, X.T) + params['coef0'], params['degree'])
    elif params['kernel'] == 'sigmoid':
        return lambda X_fit, X: np.tanh(params['gamma'] * np.dot(X_fit, X.T) + params['coef0'])
    else:
        assert False, f'want kernel in [linear, rbf, poly, sigmoid], but got {params["kernel"]}'
