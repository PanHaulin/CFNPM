import argparse
from cfnp.modules import MODULES
from cfnp.methods import METHODS, BASES
import pytorch_lightning as pl
from cfnp.utils.checkpointer import MonitorCheckpointer
import cfnp.baselines as baselines
from cfnp.args.dataset import REGRESSION_DATASETS

def parse_args_main():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--data_source", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)

    # important
    parser.add_argument("--base", type=str, required=True, choices=['base','sep','add'])
    parser.add_argument("--method", type=str, required=True, choices=['gp','klr','knn','knr','krr','svc','svc'])
    parser.add_argument("--module", type=str, required=True)

    temp_args, _ = parser.parse_known_args()
    parser = MODULES[temp_args.module].add_specific_args(parser)
    parser = METHODS[temp_args.method](BASES[temp_args.base]).add_specific_args(parser)

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    temp_args, _ = parser.parse_known_args()
    if temp_args.fast_dev_run is not None:
        parser = MonitorCheckpointer.add_specific_args(parser)

    # logger
    parser.add_argument("--api_keys", type=str)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument("--logger_project_name", type=str)
    parser.add_argument("--logger_run_name", type=str)
    parser.add_argument("--logger_description", type=str)
    parser.add_argument("--logger_tags", type=str, nargs='+', default=[])

    # baselines
    parser.add_argument('--run_baselines', action='store_true')
    temp_args, _ = parser.parse_known_args()
    if temp_args.run_baselines:
        # 设置默认basleines, 添加baselines参数
        dataset_based_baselines = getattr(baselines, 'DATASET_BASED_BASELINES_FOR_REGRESSION') if temp_args.dataset_name in REGRESSION_DATASETS else getattr(
            baselines, 'DATASET_BASED_BASELINES_FOR_CLASSIFICATION')
        if hasattr(baselines, f"{temp_args.method.upper()}_BASED_BASELINES"):
            method_base_baselines = getattr(
                baselines,  f"{temp_args.method.upper()}_BASED_BASELINES")
        baselines_dict = {**dataset_based_baselines, **method_base_baselines}
        default_bls = []
        for baseline_name in baselines_dict:
            default_bls.append(baseline_name)
        parser.add_argument("--baselines", type=str, nargs='+', default=default_bls)

        # 获取各baseline参数
        temp_args, _ = parser.parse_known_args()
        for baseline_name in temp_args.baselines:
            parser = baselines_dict[baseline_name].add_specific_args(parser)


    # evaluate
    parser.add_argument('--evaluate', action='store_true')

    # resume
    parser.add_argument('--resume', action='store_true')

    parser.add_argument("--manual_seed", type=int)

    args = parser.parse_args()
    return args
    


def search_args():
    pass

