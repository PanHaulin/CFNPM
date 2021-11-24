
import numpy as np
from argparse import ArgumentParser
import pickle
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import StratifiedShuffleSplit
from cfnp.baselines.base import RegressionBaseline


class StratifiedSampling(RegressionBaseline):
    @staticmethod
    def add_specific_args(parent_parser: ArgumentParser):
        parent_parser = super(StratifiedSampling, StratifiedSampling).add_specific_args(parent_parser)
        parser = parent_parser.add_argument_group('stratified_sampling')
        
        # specific
        parser.add_argument("--stratified_sampling_k_fold", type=int, default=5)
        parser.add_argument("--stratified_sampling_n_bins", type=int, default=5)
        parser.add_argument("--stratified_sampling_encode", type=str, default='ordinal', choices=['ordinal'])
        parser.add_argument("--stratified_sampling_sampling_strategy", type=str, default='quantile', choices=['quantile'])
        return parent_parser

    @staticmethod
    def run(MethodClass, logger, data, args):
        print('>> run baseline: Prototype Generation with ClusterCentroids')

        X_train, y_train, X_test, y_test = data

        best_model = None
        best_test_mae = -1
        best_test_mse = -1
        train_mae_list = []
        train_mse_list = []
        test_mae_list = []
        test_mse_list = []

        if args.resume:
            best_model = pickle.load(open(args.checkpoints_dir+'stratified_sampling_model.pkl'))
        else:
            
            # 数据分箱
            kb = KBinsDiscretizer(n_bins=args.stratified_sampling_n_bins, encode=args.stratified_sampling_encode, strategy=args.stratified_sampling_sampling_strategy)
            y_bin = kb.fit_transform(y_train.reshape(-1,1))
            Xy_train = np.concatenate((X_train, y_train.reshape(-1,1)), axis=1)
            # 求k次平均
            skf = StratifiedShuffleSplit(n_splits=args.stratified_sampling_k_fold, test_size=args.cmp_ratio, random_state=args.manual_seed)
            for train_idx, _ in skf.split(Xy_train, y_bin):
                # compressed by stratified sampling
                X_compressed = Xy_train[train_idx][:,:-1]
                y_compressed = Xy_train[train_idx][:,-1]

                # build model
                model = MethodClass.build_np_model(**args.__dict__)

                # train model
                model.fit(X_compressed, y_compressed)

                # eval model
                mae_train, mse_train, mae_test, mse_test = super(StratifiedSampling, StratifiedSampling).predict(
                    model=model,
                    data=(X_train, y_train, X_test, y_test)
                )

                # push to mae,mse list
                train_mae_list.append(mae_train)
                train_mse_list.append(mse_train)
                test_mae_list.append(mae_test)
                test_mse_list.append(mse_test)

                # replace best: mae和mse总提升百分比大于0
                if (best_test_mae - mae_test) / best_test_mae +  (best_test_mse - mse_test) / best_test_mse> 0:
                    best_test_mae = mae_test
                    best_test_mse = mse_test
                    best_model = model
                else:
                    del model
                
            # save model
            pickle.dump(best_model, open(args.checkpoints_dir+'stratified_sampling_model.pkl', "wb"))

        # best model inference by rapids

        best_mae_train, best_mse_train, best_mae_test, best_mse_test = super(StratifiedSampling, StratifiedSampling).predict(
            model=best_model,
            data=(X_train, y_train, X_test, y_test)
        )

        del best_model

        print('Stratified Sampling:')
        print('best_mae_train: ',best_mae_train)
        print('best_mae_test: ',best_mae_test)
        print('best_mse_train: ',best_mse_train)
        print('best_mse_test: ',best_mse_test)

        super(StratifiedSampling, StratifiedSampling).log(
            baseline_name='stratified_sampling',
            logger=logger,
            best_mae=(best_mae_train, best_mae_test),
            best_mse=(best_mse_train, best_mse_test),
            avg_mae=(np.mean(train_mae_list), np.mean(test_mae_list)),
            avg_mse=(np.mean(train_mse_list), np.mean(test_mse_list))
        )
        