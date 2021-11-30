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
# from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error as mae_score
from sklearn.metrics import mean_squared_error as mse_score
from cfnp.utils.km import get_cal_km_func_numpy, get_cal_km_func_torch
import numpy as np
from cuml.svm import SVR
from torchmetrics.functional import mean_absolute_error, mean_squared_error
from hyperopt import hp, STATUS_OK, fmin, tpe
from hyperopt.fmin import generate_trials_to_calculate
from typing import Any, Callable, Dict, List, Sequence, Tuple


def create_svr_class(base):
    '''
    动态继承父类
    '''

    class CompressionForSVR(base):
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
            self.criterion = nn.SmoothL1Loss()

            # 预测用信息
            # self.data = data

        def get_constrained_coef(self):
            alpha = F.softmax(torch.abs(self.fx_layer.weight), dim=-1)  # 行处理, (1, n_compressed)
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

            # cal mae, mse
            mae = mean_absolute_error(fx, y)
            mse = mean_squared_error(fx, y)

            if stage:
                self.log(f'pl_{stage}_loss', loss, prog_bar=True)
                self.log(f'pl_{stage}_mae', mae, prog_bar=True)
                self.log(f'pl_{stage}_mse', mse, prog_bar=True)
            
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
                CompressionForSVR, CompressionForSVR).add_specific_args(parent_parser)
            parser = parent_parser.add_argument_group('svr')

            # specific
            parser.add_argument("--kernel", type=str, required=True)
            parser.add_argument("--C", type=float, default=1.0)
            parser.add_argument("--gamma", type=float, default=None)
            parser.add_argument("--coef0", type=float, default=0.0)
            parser.add_argument("--degree", type=int, default=3)

            return parent_parser

        # 以下函数为svc独有的外部调用函数
        @staticmethod
        def build_np_model(**kwargs):
            return SVR(kernel=kwargs['kernel'], C=kwargs['C'], gamma=kwargs['gamma'], coef0=kwargs['coef0'], degree=kwargs['degree'])

        @staticmethod
        def train_original_model(logger, data, args):

            X_train, y_train, X_test, y_test = data

            # 生成存储路径
            if args.resume_checkpoints_dir == None:
                time_tick = datetime.now().strftime('%y%m%D%H%M%S')
                args.checkpoints_dir += '/{}/{}-{}-{}/'.format(
                    args.kernel, logger.version, args.logger_run_name, time_tick)
            else:
                args.checkpoints_dir += '/{}/{}/'.format(args.kernel, args.resume_checkpoints_dir)

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
                    model = CompressionForSVR.build_np_model(
                        kernel=args.kernel, C=params['C'], gamma=params['gamma'], coef0=args.coef0, degree=args.degree)
                    model.fit(X_train, y_train)
                    pred_test = model.predict(X_test)
                    loss = F.smooth_l1_loss(torch.Tensor(y_test), torch.Tensor(pred_test)).item() # 取出0-dim Tensor中的值
                    return {'loss': loss, 'status': STATUS_OK}

                trials = generate_trials_to_calculate(
                    [{'C': args.C, 'gamma': args.gamma}])
                best_params = fmin(f, space, algo=tpe.suggest,
                                   max_evals=args.max_evals, trials=trials)
                args.C = best_params['C']
                args.gamma = best_params['gamma']

                # 使用最优参数重新训练模型
                print('>> retrain original model using best params')
                model = CompressionForSVR.build_np_model(
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
            print('>> get best svr predictition by rapids ')
            pred_train_rapids = model.predict(X_train)
            pred_test_rapids = model.predict(X_test)
            mae_train_rapids = mae_score(y_train, pred_train_rapids)
            mse_train_rapids = mse_score(y_train, pred_train_rapids)
            mae_test_rapids = mae_score(y_test, pred_test_rapids)
            mse_test_rapids = mse_score(y_test, pred_test_rapids)

            print('mae_train_rapids: ', mae_train_rapids)
            print('mse_train_rapids: ', mse_train_rapids)
            print('mae_test_rapids: ', mae_test_rapids)
            print('mse_test_rapids: ', mse_test_rapids)


            # inference by params
            print('>> get best svc predictition by params ')
            X_fit = X_train[original_params['support_idx']]

            # 跳过外部预测，直接使用rapids预测结果，减少时间开销（必须保证二者结果相同）
            # pred_train_params, _, mae_train_params, mse_train_params, mae_test_params, mse_test_params = CompressionForSVR.eval_params(
            #     data, args, X_fit, original_params['coef_fit'], original_params['intercept_fit'])

            pred_train_params = pred_train_rapids
            mae_train_params = mae_train_rapids
            mse_train_params = mse_train_rapids
            mae_test_params = mae_test_rapids
            mse_test_params = mse_test_rapids

            print('mae_train_params: ', mae_train_params)
            print('mse_train_params: ', mse_train_params)
            print('mae_test_params: ', mae_test_params)
            print('mse_test_params: ', mse_test_params)

            # log, 需要保证rapids和params得到的结果是一致的
            logger.log_metrics({
                'original_mae_train_rapids': mae_train_rapids,
                'original_mse_train_rapids': mse_train_rapids,
                'original_mae_test_rapids': mae_test_rapids,
                'original_mse_test_rapids': mse_test_rapids,
                'original_mae_train_params': mae_train_params,
                'original_mse_train_params': mse_train_params,
                'original_mae_test_params': mae_test_params,
                'original_mse_test_params': mse_test_params,
            })

            return X_fit, pred_train_params[original_params['support_idx']], original_params, args

        @staticmethod
        def predict_by_params(X, X_fit, coef, intercept, args):
            cal_km = get_cal_km_func_numpy(**args.__dict__)
            km = cal_km(X_fit, X)
            fx = np.sum(km.T * coef, axis=1) + intercept
            return fx

        @staticmethod
        def eval_params(data, args, X_fit, coef, intercept):
            X_train, y_train, X_test, y_test = data

            pred_train_params = CompressionForSVR.predict_by_params(
                X_train, X_fit, coef, intercept, args)
            pred_test_params = CompressionForSVR.predict_by_params(
                X_test, X_fit, coef, intercept, args)
            mae_train_params = mae_score(y_train, pred_train_params)
            mse_train_params = mse_score(y_train, pred_train_params)
            mae_test_params = mae_score(y_test, pred_test_params)
            mse_test_params = mse_score(y_test, pred_test_params)

            return pred_train_params, pred_test_params, mae_train_params, mse_train_params, mae_test_params, mse_test_params

        def eval_compression_results(self, logger, data, args):
            # !Warning: outside和inside得到的准确率不一致，不是kernel的问题,可能是batch的问题
            compressed_X_fit, compressed_coef_fit, compressed_intercept_fit = self.get_compression_results()
            _, _, mae_train_params, mse_train_params, mae_test_params, mse_test_params = CompressionForSVR.eval_params(
                data, args, compressed_X_fit, compressed_coef_fit, compressed_intercept_fit)

            print('compression_mae_train_params: ', mae_train_params)
            print('compression_mse_train_params: ', mse_train_params)
            print('compression_mae_test_params: ', mae_test_params)
            print('compression_mse_test_params: ', mse_test_params)

            # 验证 Tensor 和 np.array的计算是否一致
            # X_train, y_train, X_test, y_test = data
            # km = self.cal_km(torch.Tensor(compressed_X_fit),torch.Tensor(X_train))
            # fx = self.cal_fx(km.T).view(-1,).detach().numpy()
            # pred_train = fx
            # acc_train = accuracy_score(y_train, pred_train)

            # km = self.cal_km(torch.Tensor(compressed_X_fit),torch.Tensor(X_test))
            # fx = self.cal_fx(km.T).view(-1,).detach().numpy()
            # pred_test = fx
            # acc_test = accuracy_score(y_test, pred_test)

            # print('torch_acc_train_params',acc_train)
            # print('torch_acc_test_params',acc_test)

            logger.log_metrics({
                'compression_mae_train_params': mae_train_params,
                'compression_mse_train_params': mse_train_params,
                'compression_mae_test_params': mae_test_params,
                'compression_mse_test_params': mse_test_params,
            })

    return CompressionForSVR
