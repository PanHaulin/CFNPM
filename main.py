from logging import error
import random

import numpy as np

from cfnp.args.setup import parse_args_main
from cfnp.methods import METHODS, BASES
# from cfnp.modules import MODULES
import cfnp.baselines as baselines
from cfnp.utils.load_data import load_data
from cfnp.utils.checkpointer import MonitorCheckpointer
from cfnp.utils.dm_factory import DataModuleFactory

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
import os
from pathlib import Path
from cfnp.args.dataset import REGRESSION_DATASETS
import json
from cfnp.utils.helper import get_max_n_ins, bit_to_str

def main():
    # 获取参数
    print("==> parsing args")
    args = parse_args_main()

    # 设置随机种子
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)
    seed_everything(args.manual_seed)

    # 获取方法类
    MethodClass = METHODS[args.method](BASES[args.base])
    # assert args.base in BASES, f"Choose base from {BASES.keys()}"
    # assert args.method in METHODS, f"Choose method from {METHODS.keys()}"
    

    # # 检查模块是否支持
    # assert args.module in MODULES, f"Choose module from {MODULES.keys()}"
    # ModuleClass = MODULES[args.module]

    # 初始化logger
    print("==> initializing logger")
    logger = NeptuneLogger(
        mode='debug' if args.fast_dev_run or args.offline else 'sync',
        api_key=args.api_keys,
        project=args.logger_project_name,
        name=args.logger_run_name,
        description=args.logger_description,
        # tags=args.logger_tags,
    )
    if args.resume and args.resume_checkpoints_dir != None:
        p = args.resume_checkpoints_dir.split("-")
        args.resume_version = p[0] + '-' +p[1]
    else:
        args.resume_version = None

    # 生成checkpoint路径
    if not args.checkpoints_dir:
        # 未设置存储路径
        args.checkpoints_dir = 'cfnp/checkpoints/{}/'.format(args.method)

    # 加载数据
    print("==> load data")
    X_train, y_train, X_test, y_test = load_data(
        args.data_source, args.dataset_name)
    
    # 创建原始机器学习模型并进行超参数调优, 并：
    # 1. 获取 X_fit, y_fit
    # 2. 返回相关参数 original_params
    # 3. 更新 args
    print("==> training original model")
    
    X_fit, y_fit, original_params, args = MethodClass.train_original_model(
        logger=logger, data=(X_train, y_train, X_test, y_test), args=args)
    
    # 根据容量大小计算最大ins限制
    if args.cmp_ratio is not None and args.limited_memory is not None:
        print('calculate n_compressed by args.cmp_ratio and args.limited_memory')
        n_compressed = int(X_fit.shape[0] * (1- args.cmp_ratio))
        max_n_ins = get_max_n_ins(args.limited_memory, percision=64, n_features=X_fit.shape[1])
        if n_compressed > max_n_ins:
            args.n_compressed = max_n_ins
        else:
            args.n_compressed = n_compressed
    elif args.limited_memory is not None:
        print('calculate n_compressed by args.limited_memory')
        args.n_compressed = get_max_n_ins(args.limited_memory, percision=64, n_features=X_fit.shape[1])
    elif args.cmp_ratio is not None:
        print('calculate n_compressed by args.cmp_ratio')
        args.n_compressed = int(X_fit.shape[0] * (1- args.cmp_ratio))
    else:
        # args.cmp_ratio is None and args.limited_memory is None
        print('calculate n_compressed by default: 20% max_memory_used')
        # 默认使用最大占用内存的20%,且
        max_data_size = X_fit.shape[0] * (X_fit.shape[1] + 1) * 64
        max_km_size = X_fit.shape[0] * X_fit.shape[0] * 64
        args.limited_memory = bit_to_str(args.limited_ratio * max(max_data_size, max_km_size), floor=True)
        args.n_compressed = get_max_n_ins(args.limited_memory, percision=64, n_features=X_fit.shape[1])

    print('cmp_ratio:{}, limited_memory:{}, n_compressed:{}'.format(args.cmp_ratio, args.limited_memory, args.n_compressed))

    # 逐个 baseline 运行
    if args.run_baselines:
        dataset_based_baselines = getattr(baselines, 'DATASET_BASED_BASELINES_FOR_REGRESSION') if args.dataset_name in REGRESSION_DATASETS else getattr(
            baselines, 'DATASET_BASED_BASELINES_FOR_CLASSIFICATION')
        if hasattr(baselines, f"{args.method.upper()}_BASED_BASELINES"):
            method_base_baselines = getattr(
                baselines,  f"{args.method.upper()}_BASED_BASELINES")

        baselines_dict = {**dataset_based_baselines, **method_base_baselines}
        print(args.baselines)
        for baseline_name in args.baselines:
            print(f"==> run baseline {baseline_name}")
            BaselineClass = baselines_dict[baseline_name]
            BaselineClass.run(
                MethodClass=MethodClass,
                logger=logger,
                data=(X_fit, y_fit, X_test, y_test),
                args=args
            )

        # for _, BaselineClass in {**dataset_based_baselines, **method_base_baselines}:
        #     print(f"==> run baseline {BaselineClass}")
        #     BaselineClass.run(
        #         MethodClass=MethodClass,
        #         logger=logger,
        #         data=(X_fit, y_fit, X_test, y_test),
        #         args=args
        #     )

    # 保存超参数
    print("==> saving args")
    logger.log_hyperparams(args)
    json_path = Path(args.checkpoints_dir) / 'args.json'
    json.dump(vars(args), open(json_path, 'w'),
              default=lambda o: '<not serializable>')

    # 准备训练压缩网络
    # 通过工厂类来建立构建具体的DataModule
    print("==> initializing datamodule")
    print('X_train: ',X_train.shape)
    print('y_train: ',X_train.shape)
    print('X_test: ',X_test.shape)
    print('y_test: ',y_test.shape)
    dm = DataModuleFactory.create_datamodule(
        type='general',
        args=args,
        X_train=X_fit,
        X_test=X_test,
        y_train=y_fit,
        y_test=y_test)

    # 模块初始化
    # print(f"==> initializing module {ModuleClass}")
    # module = ModuleClass(**args.__dict__)

    # 初始化模型
    print("==> initializing model")
    if not args.evaluate:
        model = MethodClass(X_fit=X_fit, data=(X_fit, y_fit, X_test, y_test), **original_params, **args.__dict__)
    else:
        model  = MethodClass.load_from_checkpoint(X_fit=X_fit, data=(X_fit, y_fit, X_test, y_test), **original_params, **args.__dict__)

    # 初始化 callbacks
    print("==> initializing callbacks")
    callbacks = []

    # early stopping callback
    if args.patience:
        early_stopping = EarlyStopping(
            monitor='pl_valid_acc',
            patience=args.patience,
            mode='max',
            strict=True,
        )
        callbacks.append(early_stopping)

    # checkpoint save callback
    if not args.fast_dev_run:
        # 调试时不保存模型
        ckpt = MonitorCheckpointer.get_instance(args)
        callbacks.append(ckpt)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # 初始化Trainer
    print("==> initializing trainer")
    trainer = Trainer.from_argparse_args(
        args,
        logger = logger,
        callbacks=callbacks,
        strategy=DDPPlugin(find_unused_parameters=True),
        detect_anomaly=True,
    )

    # 训练并验证
    if not args.evaluate:
        print("==> training and validating")
        if args.resume:
            # 获取pl model的ckpt文件名
            file_name = 'last.ckpt'
            # list_dir = os.listdir(args.checkpoints_dir)
            # for f in list_dir:
            #     if f.split(".")[-1] == "ckpt":
            #         file_name = f
            #         break 
            # # 读取权重进行训练
            trainer.fit(model, datamodule=dm, ckpt_path=args.checkpoints_dir + file_name)
        else:
            trainer.fit(model, datamodule=dm)

        # 用最好的模型测试
        print("==> testing best model")
        best_model_path = ckpt.best_model_path
        model  = MethodClass.load_from_checkpoint(X_fit=X_fit, data=(X_fit, y_fit, X_test, y_test), **original_params, checkpoint_path=best_model_path, **args.__dict__)
        trainer.test(model, datamodule=dm)
    else:
        # 测试
        print("==> testing loaded model")
        trainer.test(model, datamodule=dm)

    # 外部测试（depercated）
    # print("==> testing outside models")
    # model.eval_compression_results(logger=logger, data=(X_train, y_train, X_test, y_test), args=args)

    logger.finalize(status='success')

if __name__ == '__main__':
    main()