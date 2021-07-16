import os
import glob
import pandas as pd
import math
import numpy as np
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sn
from collections import defaultdict


def concat_crossval(dataset, class_names, expname, topk=1, show_heatmap=False, verbose=False):
    topk_targets = []
    topk_preds = []
    topk_f1s = []
    for k in range(1, topk + 1):
        output_dir = 'exp_results'
        epoch_file = f'top{k}_epoch_test_f1.csv'
        ensemble_epoch_file = f'{expname}_top{k}_ensembled_epochs.csv'

        root_dir = os.path.join(output_dir, f'config_{dataset}')
        count = 0
        dfs = []
        for fold in glob.glob(f'{root_dir}/{expname}*'):
            count += 1
            dfs.append(pd.read_csv(os.path.join(fold, epoch_file), index_col=0))

        assert count == 5, "something is wrong"

        df = pd.concat(dfs)
        df.reset_index()
        output_dir = os.path.join(output_dir, dataset)
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, ensemble_epoch_file))

        imgs, preds, targets = [list(df[key]) for key in ['image', 'prediction', 'target']]

        cm = confusion_matrix(targets, preds)
        accuracy = accuracy_score(targets, preds)
        f1s = [round(f1, 4) for f1 in f1_score(targets, preds, average=None)]
        f1 = sum(f1s)/len(f1s)

        topk_targets.append(targets)
        topk_preds.append(preds)
        topk_f1s.append(f1s)

        if verbose:
            print(expname, 'balanced f1: {}, overall accuracy: {}'.format(round(f1, 4), round(accuracy, 4)))
            print(cm)
            print('f1 score', f'{class_names[0]}: {f1s[0]}, '
                              f'{class_names[1]}: {f1s[1]}, '
                              f'{class_names[2]}: {f1s[2]}, '
                              f'{class_names[3]}: {f1s[3]}, '
                              f'{class_names[4]}: {f1s[4]}', '\n')

        if show_heatmap:
            num_classes = len(list(set(targets)))
            df_cm = pd.DataFrame(cm, range(num_classes), range(num_classes))
            sn.set(font_scale=1.4)
            sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g', annot_kws={"size": 16}, vmin=0, vmax=2500)  # font size
            plt.yticks(rotation=0)
            plt.title(title)
            plt.show()

    print('\n===========================================================================')
    print(f'{expname} - average results for {topk} models with 95% confidence interval')

    ba_f1s = [sum(f1s) / len(f1s) for f1s in topk_f1s]
    ba_f1_mean = sum(ba_f1s) / len(ba_f1s)
    ba_f1_std = math.sqrt(sum((x - ba_f1_mean) ** 2 for x in ba_f1s) / len(ba_f1s)) * 1.96
    print(f"overall balanced f1: {round(ba_f1_mean * 100, 2)} \u00B1 {round(ba_f1_std * 100, 2)}\n")

    for i, class_name in enumerate(class_names):
        class_f1s = [topk_f1[i] for topk_f1 in topk_f1s]
        mean = sum(class_f1s) / len(class_f1s)
        std = math.sqrt(sum((x - mean) ** 2 for x in class_f1s) / len(class_f1s)) * 1.96

        print(f"{class_name} f1: {round(mean * 100, 2) } \u00B1 {round(std * 100, 2)}")


if __name__ == '__main__':
    experiments = ['resnet50_torch', 'resnet50_bit-s', 'resnet50_bit-m']
    class_names = ['normal', 'obsolescent', 'solidified', 'disappearing', 'non-glom']
    for exp in experiments:
        concat_crossval('renal', class_names, exp, topk=5, verbose=False)


