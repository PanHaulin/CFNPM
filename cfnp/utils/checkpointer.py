import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from cfnp.args.dataset import REGRESSION_DATASETS
class MonitorCheckpointer(ModelCheckpoint):
    '''
    具有监控指标、保存topk和保存last功能的checkpointer
    '''
    def __init__(
        self,
        args: Namespace,
        logdir,
        filename,
        frequency: int = 1,
        # keep_previous_checkpoints: bool = False, #? 什么情况下需要保存之前的checkpoint？
        monitor: Optional[str] = None,
        mode: str = "min",
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
    ):
        """Custom checkpointer callback that stores checkpoints in an easier to access way.

        Args:
            args (Namespace): namespace object containing at least an attribute name.
            logdir (Union[str, Path], optional): base directory to store checkpoints.
                Defaults to "trained_models".
            frequency (int, optional): number of epochs between each checkpoint. Defaults to 1.
            keep_previous_checkpoints (bool, optional): whether to keep previous checkpoints or not.
                Defaults to False.
        """

        super().__init__(
            dirpath=logdir,
            filename= filename,
            monitor=monitor,
            mode = mode,
            save_last=save_last,
            save_top_k=save_top_k,
        )

        self.args = args
        self.logdir = Path(logdir)
        self.frequency = frequency
        self.monitor = monitor
        # self.keep_previous_checkpoints = keep_previous_checkpoints

    @staticmethod
    def add_specific_args(parent_parser: ArgumentParser):
        """Adds user-required arguments to a parser.

        Args:
            parent_parser (ArgumentParser): parser to add new args to.
        """

        parser = parent_parser.add_argument_group("checkpointer")
        parser.add_argument("--checkpoints_dir", type=str, default='checkpoints/')
        parser.add_argument("--checkpoint_frequency", default=1, type=int)
        parser.add_argument("--checkpoint_save_last", action="store_true")
        parser.add_argument("--checkpoint_save_topk", default=1, type=int)

        return parent_parser
    
    @staticmethod
    def get_instance(args):
        """
        生成实例
        """
        
        if args.dataset_name in REGRESSION_DATASETS:
            return MonitorCheckpointer(
                args,
                logdir=args.checkpoints_dir,
                filename= "{epoch:02d}-{pl_vallid_mae:.2f}-{pl_vallid_mse:.2f}-{pl_valid_loss:.2f}",
                frequency=args.checkpoint_frequency,
                monitor='pl_valid_loss',
                mode='min',
                save_last=args.checkpoint_save_last,
                save_top_k=args.checkpoint_save_topk
            )
        else:
            return MonitorCheckpointer(
                args,
                logdir=args.checkpoints_dir,
                filename= "{epoch:02d}-{pl_valid_acc:.6f}-{pl_valid_loss:.6f}",
                frequency=args.checkpoint_frequency,
                monitor='pl_valid_acc',
                mode='max',
                save_last=args.checkpoint_save_last,
                save_top_k=args.checkpoint_save_topk
            )
