import matplotlib.pyplot as plt
import os
import glob
import json
import numpy as np
from scipy import interp
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix


def auc_bar_chart(root_dir, experiments, exp_label, topk=5):
    classnames = ['normal', 'obsolescent', 'solidified', 'disappearing', 'non-glom']
    data = []
    for target, classname in enumerate(classnames):
        for exp in experiments:
            for k in range(1, topk + 1):
                df = pd.read_csv(f'{root_dir}/{exp}_top{k}_ensembled_epochs.csv')
                gts = list(df['target'])
                gts = list(map(lambda x: 1 if x == target else 0, gts))

                fpr, tpr, _ = roc_curve(gts, list(df[classname]))
                roc_auc = auc(fpr, tpr)

                data.append({
                    'classname': classname,
                    'model': exp_label[exp],
                    'auc': roc_auc
                })

    with open('plot_data/auc_bars.json', 'w') as f:
        json.dump(data, f)


def roc_curve_with_error_band(dataset, experiments, exp_label, topk=5, crossval=True):
    if dataset == 'renal':
        output_dir = 'exp_results'
        data = []
        for exp in experiments:
            root_dir = os.path.join(output_dir, f'config_{dataset}')
            fprs = []
            tprs = []
            for k in range(1, topk + 1):
                ensemble_file = f'top{k}_{exp}_ensembled_prediction.csv'
                imgs = targets = scores = None
                count = 0
                for fold in glob.glob(f'{root_dir}/{exp}*'):
                    count += 1
                    df = pd.read_csv(os.path.join(fold, f'top{k}_predict_f1.csv'))

                    imgs, targets = df['image'], df['target']

                    if scores is None:
                        scores = np.asarray(list(df['sclerosis_score']))
                    else:
                        scores += np.asarray(list(df['sclerosis_score']))

                assert count == 5, "something is wrong"

                averaged_scores = list(scores / 5)
                df_data = [[img, score, target] for img, score, target in zip(imgs, averaged_scores, targets)]
                df = pd.DataFrame(df_data, columns=['image', 'score', 'target'])
                df.to_csv(os.path.join(output_dir, ensemble_file))

                fpr, tpr, thresh = roc_curve(targets, averaged_scores, pos_label=1)
                auc_score = roc_auc_score(targets, averaged_scores)

                fprs.append(fpr)
                tprs.append(tpr)

            all_fpr = np.unique(np.concatenate([fpr for fpr in fprs]))
            all_tpr = []
            for fpr, tpr in zip(fprs, tprs):
                all_tpr.extend(np.interp(all_fpr, fpr, tpr))

            all_fpr = np.tile(all_fpr, topk)
            data.extend([{'Specificity':x, 'Sensitivity':y, 'model':exp_label[exp]} for x, y in zip(all_fpr, all_tpr)])

        with open('plot_data/roc.json', 'w') as f:
            json.dump(data, f)

    else:
        if crossval:
            output_dir = 'exp_results'
            predict_file = 'predict_acc.csv'
            ensemble_file = 'ensembled_ext_prediction.csv'

            root_dir = os.path.join(output_dir, f'config_{dataset}')

            img_history = None
            pred_history = []

            count = 0
            for fold in glob.glob(f'{root_dir}/{expname}*'):
                count += 1
                file = pd.read_csv(os.path.join(fold, predict_file))
                img_history = np.asarray(file['image'])

                pred_history.append(list(file['prediction']))

            assert count == 5, "something is wrong"

            pred_num = len(pred_history[0])
            ens_pred_history = []
            for i in range(pred_num):
                votes = []
                for j in range(5):
                    votes.append(pred_history[j][i])
                ens_pred = int(Counter(votes).most_common(1)[0][0])
                pred_entry = [0.0 for _ in range(7)]
                pred_entry[ens_pred] = 1.0
                ens_pred_history.append(pred_entry)

            output_dir = os.path.join(output_dir, dataset)
            os.makedirs(output_dir, exist_ok=True)

            data = np.concatenate((img_history[..., np.newaxis], np.stack(ens_pred_history, axis=0)), axis=1)
            df = pd.DataFrame(data, columns=['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'])
            df.loc[len(df)] = ['ISIC_0035068', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            df.to_csv(os.path.join(output_dir, expname + '_' + ensemble_file), index=False)


experiments = ['resnet50_torch', 'resnet50_bit-s', 'resnet50_bit-m']
exp_label = {'resnet50_torch': 'R50-TORCH', 'resnet50_bit-s': 'R50-BiT-S', 'resnet50_bit-m': 'R50-BiT-M',
             'resnet101_torch': 'R101-TORCH', 'resnet101_bit-s': 'R101-BiT-S', 'resnet101_bit-m': 'R101-BiT-M'}
auc_bar_chart('exp_results/renal', experiments, exp_label, topk=5)
roc_curve_with_error_band('renal', experiments, exp_label, topk=5)