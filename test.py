#%%
import numpy as np
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel, sigmoid_kernel
import torch
from torch.functional import Tensor

data_dict = np.load('temp/linear_all_data_dict.npy', allow_pickle=True).item()

compressed_X_fit = data_dict['compressed_X_fit']
compressed_coef_fit = data_dict['compressed_coef_fit']
compressed_intercept_fit = data_dict['compressed_intercept_fit']
X_train = data_dict['X_train']
X_test = data_dict['X_test']
y_train = data_dict['y_train']
y_test = data_dict['y_test']
# %%
# 验证 torch流程和numpy流程是否等价
from cfnp.utils.km import get_cal_km_func_torch, get_cal_km_func_numpy
kernel = 'linear'
C = 1.0
gamma = 1 / (X_train.shape[1] *X_train.var())
coef0 = 0.0
degree = 3

torch_km_func = get_cal_km_func_torch(kernel,gamma,coef0,degree)
numpy_km_func = get_cal_km_func_numpy(kernel,gamma,coef0,degree)

torch_km = torch_km_func(torch.Tensor(compressed_X_fit), torch.Tensor(X_train))
numpy_km = numpy_km_func(compressed_X_fit, X_train)

print(torch_km.numpy()-numpy_km)

print(torch_km.numpy()[:10])

print(numpy_km[:10])
y_pred = np.zeros(y_train.shape)
torch_fx = np.sum(torch_km.numpy().T * compressed_coef_fit, axis=1) + compressed_intercept_fit
numpy_fx = np.sum(numpy_km.T * compressed_coef_fit, axis=1) + compressed_intercept_fit
print(torch_fx)
print(numpy_fx)

tensor_fx = torch.mm(torch_km.T, torch.Tensor(compressed_coef_fit).T) + compressed_intercept_fit
print(tensor_fx.numpy())

print(numpy_fx.shape)

from sklearn.metrics import accuracy_score
torch_pred = y_pred
numpy_pred = y_pred
torch_pred[torch_fx>0]=1
numpy_pred[numpy_fx>0]=1
print(accuracy_score(y_train, torch_pred))
print(accuracy_score(y_train, numpy_pred))

print(np.sum(np.abs(compressed_coef_fit)))

import torch.nn.functional as F
alpha = F.softmax(torch.abs(torch.Tensor(compressed_coef_fit)),dim=-1)
torch.sum(alpha)
# %%
# 测试baseline输入
from sklearn.datasets import make_classification
from imblearn.under_sampling import ClusterCentroids 
X, y = make_classification(n_classes=2, class_sep=2,
weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)

# %%
# 依据coef_fit:np.array 对 X_fit: Tensor(1,1,n_samples,n_features) 进行划分
from cfnp.utils.helper import X_fit_to_tensor
from collections import Counter

X_fit = X_fit_to_tensor(compressed_X_fit)
print("X_fit size: ",X_fit.size())
coef_sign = np.sign(compressed_coef_fit)
print("coef_sign shape: ", coef_sign.shape)
counter = sorted(Counter(coef_sign.flatten()).items())
print("counter: {}, type: {}".format(counter, type(counter)))
# %%
n_nagative = counter[0][1]
n_positive = counter[1][1]
print('n_positive:{}, n_negative:{}'.format(n_positive, n_nagative))

#%%
test_tensor = torch.Tensor(compressed_X_fit)
positive_tensor = test_tensor[coef_sign>0]
negative_tensor = test_tensor[coef_sign<0]
print("size:{}, positive:{}, negative:{}".format(test_tensor.size(), positive_tensor.size(), negative_tensor.size()))
# %%
# X_fit_to_tensor可以替换为：
test_tensor = torch.Tensor(compressed_X_fit)
print(test_tensor.size())
test_tensor = test_tensor.view(1, test_tensor.size(1), test_tensor.size(0), 1)
print((X_fit==test_tensor).all())
# %%
