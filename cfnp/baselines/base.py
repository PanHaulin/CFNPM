import numpy as np
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error



class ClasscificationBaseline():
    @staticmethod
    def add_specific_args(parent_parser: ArgumentParser):
        # parser = parent_parser.add_argument_group('cls_baseline_base')
        # parser.add_argument("--base_k_fold", type=int, default=5)
        # parser.add_argument("--base_sampling_strategy", type=str, default='maintain')
        return parent_parser
    
    @staticmethod
    def predict(model, data):
        X_train, y_train, X_test, y_test = data

        # inference by rapids
        pred_train_rapids = model.predict(X_train)
        pred_test_rapids = model.predict(X_test)
        acc_train_rapids = accuracy_score(y_train, pred_train_rapids)
        acc_test_rapids = accuracy_score(y_test, pred_test_rapids)

        return acc_train_rapids, acc_test_rapids

    @staticmethod
    def log(baseline_name, logger, best_acc, avg_acc):
        best_acc_train, best_acc_test = best_acc
        avg_acc_train, avg_acc_test = avg_acc

        # log
        logger.log_metrics({
            f'{baseline_name}_best_acc_train': best_acc_train,
            f'{baseline_name}_best_acc_test': best_acc_test,
            f'{baseline_name}_avg_acc_train': np.mean(avg_acc_train),
            f'{baseline_name}_avg_acc_test': np.mean(avg_acc_test)
        })

class RegressionBaseline():
    @staticmethod
    def add_specific_args(parent_parser: ArgumentParser):
        # parser = parent_parser.add_argument_group('cls_baseline_base')
        # parser.add_argument("--base_k_fold", type=int, default=5)
        return parent_parser

    @staticmethod
    def predict(model, data):
        X_train, y_train, X_test, y_test = data

        # inference by rapids
        pred_train_rapids = model.predict(X_train)
        mae_train_rapids = mean_absolute_error(y_train, pred_train_rapids)
        mse_train_rapids = mean_squared_error(y_train, pred_train_rapids)
        pred_test_rapids = model.predict(X_test)
        mae_test_rapids = mean_absolute_error(y_test, pred_test_rapids)
        mse_test_rapids = mean_squared_error(y_test, pred_test_rapids)

        return mae_train_rapids, mse_train_rapids, mae_test_rapids, mse_test_rapids

    @staticmethod
    def log(baseline_name, logger, best_mae, best_mse, avg_mae, avg_mse):

        best_mae_train, best_mae_test = best_mae
        best_mse_train, best_mse_test = best_mse
        avg_mae_train, avg_mae_test = avg_mae
        avg_mse_train, avg_mse_test = avg_mse

        logger.log_metrics({
            f'{baseline_name}_best_mae_train': best_mae_train,
            f'{baseline_name}_best_mse_train': best_mse_train,
            f'{baseline_name}_best_mae_test': best_mae_test,
            f'{baseline_name}_best_mse_test': best_mse_test,
            f'{baseline_name}_avg_mae_train': avg_mae_train,
            f'{baseline_name}_avg_mse_train': avg_mse_train,
            f'{baseline_name}_avg_mae_test': avg_mae_test,
            f'{baseline_name}_avg_mse_test': avg_mse_test
        })
