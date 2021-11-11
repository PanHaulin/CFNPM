import pytorch_lightning as pl
from cfnp.modules.conv import ConvCompressionModule
from cfnp.modules.gumble import GumbleCompressionModule
from typing import Any, Callable, Dict, List, Sequence, Tuple
import torch
from argparse import ArgumentParser
from cfnp.utils.helper import X_fit_to_tensor
import numpy as np

class BaseModel(pl.LightningModule):
    def __init__(
        self,
        X_fit: np.array,
        cmp_ratio: float,
        module: str,
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        lr: float,
        weight_decay: float,
        momentum: float,
        scheduler: str,
        **kwargs
    ):
        super().__init__()

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler

        X_fit = X_fit_to_tensor(X_fit)
        self.register_buffer('X_fit', X_fit)
        self.n_fit = X_fit.size(2)
        self.n_features = X_fit.size(1)
        self.n_compressed = round(self.n_fit * (1- cmp_ratio))
        self.cmp_ratio = cmp_ratio

        self.extra_kwargs = kwargs

        # 初始化压缩模块
        if module == 'conv':
            self.module = ConvCompressionModule()
        else:
            self.module = GumbleCompressionModule()
    
    @staticmethod
    def add_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group('cls_base')

        # general
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=8)

        # compression
        parser.add_argument("cmp_ratio", type=float, required=True)

        # hyper-opt
        parser.add_argument("--max_eval", type=int, default=100)

        return parent_parser
    
    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        return [
            {"name": "module", "params": self.module.parameters()},
        ]

    def configure_optimizers(self) -> Tuple[List, List]:
        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.learnable_params,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=True
                )
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.learnable_params, 
                lr=self.lr, 
                betas=(0.9, 0.999), 
                eps=1e-08
                )
        else:
            assert False, f"{self.optimizer} not in (sgd, adam)"

        scheduler_dict = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs),
                "interval": "epoch"
                }
        return {'optimizer':optimizer, 'lr_scheduler':scheduler_dict}
    
    def forward(self):
        return self.module(self.X_fit)

    def on_train_end(self):
        self.logger.log_metrics({
            'actual_ratio': 1-(self.n_compressed-self.n_fit),
            'before_n_ins': self.n_fit,
            'after_n_ins': self.n_compressed
        })
    
    @staticmethod
    def build_original_model():
        pass
    
    @staticmethod
    def predict_by_params():
        pass

class ClassSeparationBaseModel(BaseModel):
    '''
    类分离，正负类各自输入到各自的module中
    init中对输入进行类分离
    forward中对输出进行合并
    '''
    
    def forward(self):
        X_compression = self.module(self.X_fit)
        X_positive_compression = self.module1(self.X_positive)
        X_negative_compression = self.module2(self.X_negative)
        X_compression = torch 

class AddInstanceBaseModel(BaseModel):
    '''
    增加一些实例到压缩实例集合中
    '''
    pass