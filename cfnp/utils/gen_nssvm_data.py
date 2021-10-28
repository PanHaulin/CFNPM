import argparse
import os
import sys
from os import mkdir

sys.path.append('.')

from configs.defaults import get_cfg_defaults
from utils.data_processor import get_libsvm_data
import scipy.io as scio
import numpy as np

def generate(args, datasets, kernels):
    # command line params
    dataset_name = args.dataset.lower()
    model_type = 'svc'
    
    # dataset params
    n_features = datasets[dataset_name]['n_features']
    is_multi = datasets[dataset_name]['is_multi']
    has_test = datasets[dataset_name]['has_test']

    # load data
    X_train, y_train, X_test, y_test = get_libsvm_data(dataset_name, n_features, is_multi, has_test)

    # 修改为double类型，转置y
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    y_train = y_train.reshape(-1,1).astype(np.float64)
    y_test = y_test.reshape(-1,1).astype(np.float64)

    mat_dict = {'X': X_train, 'y': y_train, 'tX': X_test, 'ty': y_test}
    
    for kernel in kernels:
        # load info to get sv_index
        info_path = 'checkpoints/np_checkpoints/'+ dataset_name + '_' + kernel + '_' + model_type + '_info.npy'
        info = np.load(info_path, allow_pickle=True).item()
        sv_index = info['support_index']
        X_key = kernel + '_X_sv'
        y_key = kernel + '_y_sv'
        mat_dict[X_key] = X_train[sv_index]
        mat_dict[y_key] = y_train[sv_index]

    save_path = 'baselines/NSSVM/solver/realdata/mat/' + dataset_name + '.mat'

    # save data
    scio.savemat(save_path, mat_dict)
    return


def main():
    parser = argparse.ArgumentParser(description="Training non-parametric model")
    parser.add_argument("--dataset", help="a dataset name declared in config file", type=str)
    parser.add_argument("--model_type", help="np model type like: svc, lr..", type=str)
    parser.add_argument("--kernel", help="kernel like: linear, rbf, sigmoid, poly", type=str)

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    datasets = cfg.GLOBAL.DATASETS[0]
    kernels = cfg.GLOBAL.KERNELS

    # Valid Check
    if args.dataset is None:
        print('Must contain --dataset, --model_type, --kernel')
        return
    if args.dataset.lower() not in datasets:
        print('Failed: Dataset name {} not in config DATASETS!'.format(args.dataset.lower()))
        return

    generate(args, datasets, kernels)
    return


if __name__ == '__main__':
    main()