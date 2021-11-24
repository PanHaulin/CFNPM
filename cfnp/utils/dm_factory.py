from logging import info
from os import replace
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import torch.utils.data as Data
import torch
from typing import Optional
import numpy as np
from argparse import ArgumentParser

class DataModuleFactory():
    @staticmethod
    def create_datamodule(type, args, **kwargs):
        support_types = ['general']
        if type == 'general':
            return GeneralDataModule(kwargs['X_train'], kwargs['X_test'],kwargs['y_train'],kwargs['y_test'], **args.__dict__)
        else:
            assert False, f"Got datamodule {type}, but want {support_types}"


class GeneralDataModule(pl.LightningDataModule):
    '''
    读取数据用于预测
    '''
    def __init__(self, X_train, X_test, y_train, y_test, batch_size, num_workers, manual_seed, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.manual_seed = manual_seed

        self.X_train = torch.Tensor(X_train)
        self.y_train = torch.Tensor(y_train)
        self.X_test = torch.Tensor(X_test)
        self.y_test = torch.Tensor(y_test)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        
        # setup train data
        if stage == 'fit' or stage is None:
            self.trainset = Data.TensorDataset(self.X_train, self.y_train)
            self.validset = Data.TensorDataset(self.X_test, self.y_test)
            del self.X_train
            del self.y_train

        # setup test data
        if stage == 'test' or stage is None: 
            self.testset = Data.TensorDataset(self.X_test, self.y_test)
            del self.X_test
            del self.y_test
        

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, worker_init_fn=self.worker_init_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validset, batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=self.worker_init_fn)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.validset, batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        np.random.seed(int(self.manual_seed))

