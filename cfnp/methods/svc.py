from argparse import ArgumentParser
from datetime import datetime
import os
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import warnings
import pickle
from sklearn.metrics import accuracy_score
from cfnp.utils.km import get_cal_km_func_numpy, get_cal_km_func_torch
import numpy as np
from cuml.svm import SVC
from torchmetrics.functional import accuracy
from hyperopt import hp, STATUS_OK, fmin, tpe
from hyperopt.fmin import generate_trials_to_calculate
from typing import Any, Callable, Dict, List, Sequence, Tuple


def create_svc_class(base):
    '''
    动态继承父类
    '''

    class CompressionForSVC(base):
        def __init__(self, coef_fit, intercept_fit, **kwargs):
            super().__init__(coef_fit=coef_fit, **kwargs)

            # weights(n_compressed, 1), bias(1,)
            self.fx_layer = nn.Linear(self.n_compressed, 1, bias=True)

            # 用coef_fit和intercept初始化
            self.fx_layer.weight = nn.Parameter(torch.Tensor(
                (coef_fit.reshape(-1, 1))[:self.n_compressed].reshape(1, -1)))
            self.fx_layer.bias = nn.Parameter(torch.Tensor([intercept_fit]))

            # cal km func
            self.cal_km = get_cal_km_func_torch(**self.extra_kwargs)

            # criterion
            self.criterion = nn.BCEWithLogitsLoss()

            # 统计信息
            self.n_epochs = 0

            # 预测用信息
            # self.data = data

        def get_constrained_coef(self):
            alpha = F.softmax(torch.abs(self.fx_layer.weight),
                              dim=-1)  # 行处理, (1, n_compressed)
            # (1, n_compressed)
            coef = alpha * torch.sign(self.fx_layer.weight)
            return coef

        def cal_fx(self, km):
            coef = self.get_constrained_coef()
            fx = torch.mm(km, coef.T) + self.fx_layer.bias  # (batch_size, 1)
            return fx
            
        def get_compression_results(self) -> Tuple[np.array, np.array, np.array]:
            compressed_X_fit = super().forward().cpu().detach().numpy()
            compressed_coef_fit = self.get_constrained_coef().data.cpu().detach().numpy()
            compressed_intercept_fit = self.fx_layer.bias.data.cpu().detach().numpy()[0]

            return compressed_X_fit, compressed_coef_fit, compressed_intercept_fit

            # return {'X_fit':compressed_X_fit, 'coef':compressed_coef_fit, 'intercept':compressed_intercept_fit}

        @property
        def learnable_params(self) -> List[Dict[str, Any]]:
            extra_learnable_params =[
                {"name":"fx_layer", "params": self.fx_layer.parameters()}
            ]

            return super().learnable_params + extra_learnable_params


        def _shared_step(self, X, y, stage=None):

            # compressing
            compressed_X_fit = super().forward()

            # cal km
            km = self.cal_km(compressed_X_fit, X) # shape: (n_compressed, batch_size)

            # cal fx
            # 由于linear.weight是(1,n_compressed)，计算xA^T + b，因此km要转置
            # fx的维度为(batch_size, n_compressed) @ (n_compressed, 1) = (batch_size , 1)
            fx = self.cal_fx(km.T)
            del km

            # cal loss
            fx = fx.view(-1,)
            loss = self.criterion(fx, y)

            # cal acc
            with torch.no_grad():
                y_pred = torch.zeros_like(y)
                y_pred[fx > 0] = 1
                acc = accuracy(y_pred.long(), y.long())

            if stage:
                self.log(f'pl_{stage}_loss', loss, prog_bar=True)
                self.log(f'pl_{stage}_acc', acc, prog_bar=True)
            
            return loss

        def training_step(self, batch, batch_idx):

            X, y = batch
            loss = self._shared_step(X, y, 'train')
            return loss

        def validation_step(self, batch, batch_idx):
            X, y = batch
            self._shared_step(X, y, 'valid')
        
        def test_step(self, batch, batch_idx):
            X, y = batch
            self._shared_step(X, y, 'test')

        @staticmethod
        def add_specific_args(parent_parser: ArgumentParser):
            parent_parser = super(
                CompressionForSVC, CompressionForSVC).add_specific_args(parent_parser)
            parser = parent_parser.add_argument_group('svc')

            # specific
            parser.add_argument("--kernel", type=str, required=True)
            parser.add_argument("--C", type=float, default=1.0)
            parser.add_argument("--gamma", type=float, default=None)
            parser.add_argument("--coef0", type=float, default=0.0)
            parser.add_argument("--degree", type=int, default=3)

            return parent_parser

        @staticmethod
        def build_np_model(**kwargs):
            return SVC(kernel=kwargs['kernel'], C=kwargs['C'], gamma=kwargs['gamma'], coef0=kwargs['coef0'], degree=kwargs['degree'])

        @staticmethod
        def train_original_model(logger, data, args):

            X_train, y_train, X_test, y_test = data

            # 生成存储路径
            time_tick = datetime.now().strftime('%y%m%D%H%M%S')
            args.checkpoints_dir += '/{}/{}-{}-{}/'.format(
                args.kernel, logger.version, args.logger_run_name, time_tick)

            if Path(args.checkpoints_dir).is_dir():
                if args.resume:
                    # 目录存在且指定重用
                    print('>> Load exist orignal model')
                    model = pickle.load(
                        open(args.checkpoints_dir+'original_model.pkl'))
            else:
                # 指定的目录不存在
                if args.resume:
                    assert False, f'Can not resume from non-existent dir {args.checkpoints_dir}'
                print('>> Path {} does not exist, create'.format(
                    args.checkpoints_dir))
                os.makedirs(args.checkpoints_dir, exist_ok=True)

                # 需要训练原模型
                # 设置默认值
                n_features = X_train.shape[0]

                if args.gamma is None:
                    args.gamma = 1 / (n_features * X_train.var())

                # 设置搜索空间
                space = {
                    'C': hp.uniform('C', 0.1, 100),
                    'gamma': hp.uniform('gamma', 1e-3, 1)
                }

                # 参数自动调优
                print('>> hyper-params optimization')

                def f(params):
                    model = CompressionForSVC.build_np_model(
                        kernel=args.kernel, C=params['C'], gamma=params['gamma'], coef0=args.coef0, degree=args.degree)
                    model.fit(X_train, y_train)
                    pred_test = model.predict(X_test)
                    acc = accuracy_score(y_test, pred_test)
                    return {'loss': -acc, 'status': STATUS_OK}

                trials = generate_trials_to_calculate(
                    [{'C': args.C, 'gamma': args.gamma}])
                best_params = fmin(f, space, algo=tpe.suggest,
                                   max_evals=args.max_evals, trials=trials)
                args.C = best_params['C']
                args.gamma = best_params['gamma']

                # 使用最优参数重新训练模型
                print('>> retrain original model using best params')
                model = CompressionForSVC.build_np_model(
                    kernel=args.kernel, C=args.C, gamma=args.gamma, coef0=args.coef0, degree=args.degree)
                model.fit(X_train, y_train)

                # 保存模型
                pickle.dump(model, open(args.checkpoints_dir +
                            'original_model.pkl', "wb"))

            # 获取预测相关参数
            original_params = {
                # 'config': model.get_params(),
                'support_idx': model.support_,
                'coef_fit': model.dual_coef_[0],
                'intercept_fit': model.intercept_
            }

            # inference by rapids
            print('>> get best svc predictition by rapids ')
            pred_train_rapids = model.predict(X_train)
            pred_test_rapids = model.predict(X_test)
            acc_train_rapids = accuracy_score(y_train, pred_train_rapids)
            acc_test_rapids = accuracy_score(y_test, pred_test_rapids)
            print('acc_train_rapids:',acc_train_rapids)
            print('acc_test_rapids', acc_test_rapids)

            # inference by params
            print('>> get best svc predictition by params ')
            X_fit = X_train[original_params['support_idx']]
            # 跳过
            # pred_train_params, _, acc_train_params, acc_test_params = CompressionForSVC.eval_params(
            #     data, args, X_fit, original_params['coef_fit'], original_params['intercept_fit'])
            pred_train_params = pred_train_rapids
            acc_train_params = acc_train_rapids
            acc_test_params = acc_test_rapids
            print('acc_train_params:', acc_train_params)
            print('acc_test_params:', acc_test_params)

            # log, 需要保证rapids和params得到的结果是一致的
            logger.log_metrics({
                'original_acc_train_rapids': acc_train_rapids,
                'original_acc_test_rapids': acc_test_rapids,
                'original_acc_train_params': acc_train_params,
                'original_acc_test_params': acc_test_params
            })

            return X_fit, pred_train_params[original_params['support_idx']], original_params, args

        @staticmethod
        def predict_by_params(X, y_shape, X_fit, coef, intercept, args):
            cal_km = get_cal_km_func_numpy(**args.__dict__)
            y_pred = np.zeros(y_shape)
            km = cal_km(X_fit, X)
            fx = np.sum(km.T * coef, axis=1) + intercept
            y_pred[fx > 0] = 1
            return y_pred

        @staticmethod
        def eval_params(data, args, X_fit, coef, intercept):
            X_train, y_train, X_test, y_test = data

            pred_train_params = CompressionForSVC.predict_by_params(
                X_train, y_train.shape, X_fit, coef, intercept, args)
            pred_test_params = CompressionForSVC.predict_by_params(
                X_test, y_test.shape, X_fit, coef, intercept, args)
            acc_train_params = accuracy_score(y_train, pred_train_params)
            acc_test_params = accuracy_score(y_test, pred_test_params)

            return pred_train_params, pred_test_params, acc_train_params, acc_test_params

        def eval_compression_results(self, logger, data, args):
            # !Warning: outside和inside得到的准确率不一致，不是kernel的问题,是batch的问题
            compressed_X_fit, compressed_coef_fit, compressed_intercept_fit = self.get_compression_results()
            _, _, acc_train_params, acc_test_params = CompressionForSVC.eval_params(
                data, args, compressed_X_fit, compressed_coef_fit, compressed_intercept_fit)

            print('np_acc_train_params:', acc_train_params)
            print('np_acc_test_params:', acc_test_params)

            # 验证 Tensor 和 np.array的计算是否一致
            # X_train, y_train, X_test, y_test = data
            # km = self.cal_km(torch.Tensor(compressed_X_fit),torch.Tensor(X_train))
            # fx = self.cal_fx(km.T).view(-1,).detach().numpy()
            # pred_train = np.zeros(y_train.shape)
            # pred_train[fx>0]=1
            # acc_train = accuracy_score(y_train, pred_train)

            # km = self.cal_km(torch.Tensor(compressed_X_fit),torch.Tensor(X_test))
            # fx = self.cal_fx(km.T).view(-1,).detach().numpy()
            # pred_test = np.zeros(y_test.shape)
            # pred_test[fx>0]=1
            # acc_test = accuracy_score(y_test, pred_test)

            # print('torch_acc_train_params',acc_train)
            # print('torch_acc_test_params',acc_test)

            logger.log_metrics({
                'compression_acc_train_params': acc_train_params,
                'compression_acc_test_params': acc_test_params
            })

    return CompressionForSVC
