import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import time
from cfnp.args.dataset import BINARYCLASS_DATASETS, MULTICLASS_DATASETS, REGRESSION_DATASETS
from collections import Counter
from cfnp.utils.helper import multi_to_binary

def load_data(data_source, dataset_name):
    support_source = ['libsvm']
    assert data_source in support_source, f"Choose dataset from {support_source}"
    if data_source == 'libsvm':
        X_train, y_train, X_test, y_test = _load_libsvm_dataset(dataset_name)
    
    return X_train, y_train, X_test, y_test


def _load_libsvm_dataset(dataset_name, test_size=0.3):

    if dataset_name in BINARYCLASS_DATASETS:
        train_path = 'datasets/libsvm/binary/' + dataset_name
        test_path = 'datasets/libsvm/binary/' + dataset_name + '.t'
        n_features = BINARYCLASS_DATASETS[dataset_name]['n_features']
    elif dataset_name in MULTICLASS_DATASETS:
        train_path = 'datasets/libsvm/multiple/' + dataset_name
        test_path = 'datasets/libsvm/multiple/' + dataset_name + '.t'
        n_features = MULTICLASS_DATASETS[dataset_name]['n_features']
    elif dataset_name in REGRESSION_DATASETS:
        train_path = 'datasets/libsvm/regression/' + dataset_name
        test_path = 'datasets/libsvm/regression/' + dataset_name +'.t'
        n_features = REGRESSION_DATASETS[dataset_name]['n_features']
    else:
        assert False, f"Can not find {dataset_name} in BINARYCLASS_DATASETS, MULTICLASS_DATASETS and REGRESSION"
    
    if not os.path.exists(test_path):
        print('test file is not exists, split the file')
        X, y = load_svmlight_file(train_path, n_features=n_features)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        dump_svmlight_file(X_train, y_train, train_path)
        dump_svmlight_file(X_test, y_test, test_path)
        print('split and save finish')
    else:
        print('test file exists')

    # load data
    start = time.time()
    if dataset_name in BINARYCLASS_DATASETS:
        # binary classification
        X_train, y_train, X_test, y_test = _process_libsvm_data(train_path, test_path, n_features, is_classification=True, is_multi=False)
    elif dataset_name in MULTICLASS_DATASETS:
        X_train, y_train, X_test, y_test = _process_libsvm_data(train_path, test_path, n_features, is_classification=True, is_multi=True)
    else:
        # regression
        X_train, y_train, X_test, y_test = _process_libsvm_data(train_path, test_path, n_features, is_classification=False, is_multi=False)

    end = time.time()
    load_time = end - start

    print('<<<< Load dataset {}'.format(dataset_name))
    print('time used {:.2f}'.format(load_time))
    print('n_features:{}'.format(n_features))
    print('numbers:\n Train:{}, Test:{}'.format(X_train.shape[0],X_test.shape[0]))
    if dataset_name in {**BINARYCLASS_DATASETS, **MULTICLASS_DATASETS}:
        print('distribution:')
        print('trainset - {}'.format(sorted(Counter(y_train).items())))
        print('testset - {}'.format(sorted(Counter(y_test).items())))

    return X_train, y_train, X_test, y_test


def _process_libsvm_data(train_path, test_path, n_features, is_classification, is_multi):
    encoder = LabelEncoder()
    # load trainset
    X_train, y_train = load_svmlight_file(train_path, n_features=n_features)
    df = pd.DataFrame(X_train.toarray())

    for i in range(n_features):
        if str(df[i].dtype) == 'object':
            df[i] = encoder.fit_transform(df[i])
    X_train = df.values

    # load testset
    X_test, y_test = load_svmlight_file(test_path, n_features=n_features)
    df = pd.DataFrame(X_test.toarray())
    for i in range(n_features):
        if str(df[i].dtype) == 'object':
            df[i] = encoder.fit_transform(df[i])
    X_test = df.values

    if is_classification:
        # need to encode y
        if is_multi:
            # multi to binary
            labels_true, labels_false = multi_to_binary(y_train)
            for label in labels_true:
                y_train[y_train==label] = 3.14159
                y_test[y_test==label] = 3.14159
            for label in labels_false:
                y_test[y_test==label] = 0.618
                y_train[y_train==label] = 0.618

        # re-encode
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.fit_transform(y_test)

    return X_train, y_train, X_test, y_test