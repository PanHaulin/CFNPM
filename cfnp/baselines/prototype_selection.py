
import numpy as np
import pickle
from argparse import ArgumentParser
from collections import Counter
from imblearn.under_sampling import NearMiss

class PrototypeSelection:
    @staticmethod
    def add_method_specific_args(parent_parser: ArgumentParser):
        parent_parser = super(PrototypeSelection, PrototypeSelection).add_model_specific_args(parent_parser)
        return parent_parser

    @staticmethod
    def run(MethodClass, logger, data, args):
        print('>> run baseline: Prototype Selection with NearMiss')

        X_train, y_train, X_test, y_test = data

        best_model = None
        best_test_acc = -1
        train_acc_list = []
        test_acc_list = []

        # 计算压缩后样本数量
        label_distribution = sorted(Counter(y_train).items())
        n_negative = label_distribution[0][1]
        n_positive = label_distribution[1][1]
        n_compressed = int((1 - args.cmp_ratio) * (n_negative+ n_positive))

        # 计算正负样本数
        assert args.sampling_strategy in ['maintain', 'balance']
        if args.sampling_strategy == 'maintain':
            # 保持分布不变
            n_negative_compressed = n_compressed * (n_negative / (n_negative + n_positive))
            n_positive_compressed = n_compressed * (n_positive / (n_negative + n_positive))
        else:
            # 平衡样本数量
            if n_negative < int(0.5 * n_compressed):
                n_negative_compressed = n_negative
                n_positive_compressed = n_compressed - n_negative_compressed
            elif n_positive < int(0.5 * n_compressed):
                n_positive_compressed = n_positive
                n_negative_compressed = n_compressed - n_positive_compressed
            else:
                n_negative_compressed = int(0.5 * n_compressed)
                n_positive_compressed = n_compressed - n_negative_compressed

        if args.resume:
            best_model = pickle.load(open(args.checkpoints_dir+'prototype_selection_model.pkl'))
        else:
            # 求NearMiss 1-3 的平均
            for i in range(1,3):
                # compressed by prototype selection
                cc = NearMiss(sampling_strategy = {0:n_negative_compressed, 1:n_positive_compressed}, version=i)
                X_compressed, y_compressed = cc.fit(X_train, y_train)

                # build model
                model = MethodClass.build_np_models(args.__dict__)

                # train model
                model.fit(X_compressed, y_compressed)

                # eval model
                acc_train, acc_test = super(PrototypeSelection, PrototypeSelection).predict(
                    model=model,
                    data=(X_train, y_train, X_test, y_test)
                )

                # push to acc list
                train_acc_list.append(acc_train)
                test_acc_list.append(acc_test)

                # replace best
                if acc_test > best_test_acc:
                    best_test_acc = acc_test
                    best_model = model
                else:
                    del model
                
                
            # save model
            pickle.dump(best_model, open(args.checkpoints_dir+'prototype_selection_model.pkl', "wb"))

        # best model predict and log
        best_acc_train, best_acc_test = super(PrototypeSelection, PrototypeSelection).predict(
            model=best_model,
            data=(X_train, y_train, X_test, y_test),
        )

        super(PrototypeSelection, PrototypeSelection).log(
            baseline_name='prototype_selection',
            logger=logger,
            best_acc=(best_acc_train, best_acc_test),
            avg_acc=(np.mean(train_acc_list), np.mean(test_acc_list))
        )