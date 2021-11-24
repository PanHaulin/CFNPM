#%%
import numpy as np
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel, sigmoid_kernel
import torch
# %%
print('ho')
# %%
data_dict = np.load('temp/linear_all_data_dict.npy', allow_pickle=True).item()
# %%
compressed_X_fit = data_dict['compressed_X_fit']
compressed_coef_fit = data_dict['compressed_coef_fit']
compressed_intercept_fit = data_dict['compressed_intercept_fit']
X_train = data_dict['X_train']
X_test = data_dict['X_test']
y_train = data_dict['y_train']
y_test = data_dict['y_test']
# %%
from cfnp.utils.km import get_cal_km_func_torch, get_cal_km_func_numpy
# %%
kernel = 'linear'
C = 1.0
gamma = 1 / (X_train.shape[1] *X_train.var())
coef0 = 0.0
degree = 3
# %%
torch_km_func = get_cal_km_func_torch(kernel,gamma,coef0,degree)
numpy_km_func = get_cal_km_func_numpy(kernel,gamma,coef0,degree)
# %%
torch_km = torch_km_func(torch.Tensor(compressed_X_fit), torch.Tensor(X_train))
numpy_km = numpy_km_func(compressed_X_fit, X_train)
# %%
print(torch_km.numpy()-numpy_km)
# %%
print(torch_km.numpy()[:10])
# %%
print(numpy_km[:10])
# %%
y_pred = np.zeros(y_train.shape)
torch_fx = np.sum(torch_km.numpy().T * compressed_coef_fit, axis=1) + compressed_intercept_fit
numpy_fx = np.sum(numpy_km.T * compressed_coef_fit, axis=1) + compressed_intercept_fit
print(torch_fx)
print(numpy_fx)
# %%
tensor_fx = torch.mm(torch_km.T, torch.Tensor(compressed_coef_fit).T) + compressed_intercept_fit
print(tensor_fx.numpy())
# %%
print(numpy_fx.shape)
# %%
from sklearn.metrics import accuracy_score
torch_pred = y_pred
numpy_pred = y_pred
torch_pred[torch_fx>0]=1
numpy_pred[numpy_fx>0]=1
print(accuracy_score(y_train, torch_pred))
print(accuracy_score(y_train, numpy_pred))
# %%
print(np.sum(np.abs(compressed_coef_fit)))
# %%
import torch.nn.functional as F
alpha = F.softmax(torch.abs(torch.Tensor(compressed_coef_fit)),dim=-1)
torch.sum(alpha)
# %%
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import ClusterCentroids 
X, y = make_classification(n_classes=2, class_sep=2,
weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
# %%
