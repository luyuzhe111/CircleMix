import os
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


def concat_crossval(dataset, class_names, expname, topk=1, show_heatmap=False, binary_result=False, verbose=False):
    topk_targets = []
    topk_preds = []
    topk_f1s = []
    for k in range(1, topk + 1):
        output_dir = 'exp_results'
        epoch_file = f'top{k}_epoch_test_f1.csv'
        ensemble_epoch_file = f'{expname}_top{k}_ensembled_epochs.csv'

        root_dir = os.path.join(output_dir, f'config_{dataset}')
        imgs = []
        preds = []
        targets = []
        count = 0
        for dir in os.listdir(root_dir):
            if expname in dir:
                count += 1
                file = pd.read_csv(os.path.join(root_dir, dir, epoch_file))
                imgs += list(file['image'])
                preds += list(file['prediction'])
                targets += list(file['target'])

        assert count == 5, "something is wrong"

        topk_targets.append(targets)
        topk_preds.append(preds)

        data = [[img, pred, target] for img, pred, target in zip(imgs, preds, targets)]
        df = pd.DataFrame(data, columns=['image', 'prediction', 'target'])
        output_dir = os.path.join(output_dir, dataset)
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, ensemble_epoch_file))

        cm = confusion_matrix(targets, preds)
        accuracy = accuracy_score(targets, preds)
        f1s = [round(f1, 4) for f1 in f1_score(targets, preds, average=None)]
        f1 = sum(f1s)/len(f1s)

        topk_f1s.append(f1s)

        if verbose:
            print(expname, 'balanced f1: {}, overall accuracy: {}'.format(round(f1, 4), round(accuracy, 4)))
            print(cm)
            print('f1 score', f'{class_names[0]}: {f1s[0]}, '
                              f'{class_names[1]}: {f1s[1]}, '
                              f'{class_names[2]}: {f1s[2]}, '
                              f'{class_names[3]}: {f1s[3]}, '
                              f'{class_names[4]}: {f1s[4]}', '\n')

        if binary_result:
            bi_targets = list(map(lambda x: 1 if x != 4 else 0, targets))
            bi_preds = list(map(lambda x: 1 if x != 4 else 0, preds))

            acc = round(accuracy_score(bi_targets, bi_preds), 4)
            f1 = round(f1_score(bi_targets, bi_preds), 4)
            print(expname, 'binary overall accuracy: {}, glomeruli f1: {}'.format(acc, f1, '\n'))
            print(confusion_matrix(bi_targets, bi_preds))

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


def ens_prediction(dataset, expname, crossval=True):
    if dataset == 'renal':
        output_dir = 'exp_results'
        predict_file = 'predict_f1.csv'
        ensemble_file = f'{expname}_ensembled_prediction.csv'

        root_dir = os.path.join(output_dir, f'config_{dataset}')
        imgs = targets = scores = None
        count = 0
        for dir in os.listdir(root_dir):
            if expname in dir:
                count += 1
                file = pd.read_csv(os.path.join(root_dir, dir, predict_file))

                imgs = file['image']
                targets = file['target']

                if scores is None:
                    scores = np.asarray(list(file['sclerosis_score']))
                else:
                    scores += np.asarray(list(file['sclerosis_score']))

        assert count == 5, "something is wrong"

        averaged_scores = list(scores / 5)
        data = [[img, score, target] for img, score, target in zip(imgs, averaged_scores, targets)]
        df = pd.DataFrame(data, columns=['image', 'score', 'target'])
        df.to_csv(os.path.join(output_dir, ensemble_file))

        fpr, tpr, thresh = roc_curve(targets, averaged_scores, pos_label=1)
        auc_score = roc_auc_score(targets, averaged_scores)

        return fpr, tpr, thresh, round(auc_score, 3)

    else:
        if crossval:
            output_dir = 'exp_results'
            predict_file = 'predict_acc.csv'
            ensemble_file = 'ensembled_ext_prediction.csv'

            root_dir = os.path.join(output_dir, f'config_{dataset}')

            img_history = None
            pred_history = []

            count = 0
            for dir in os.listdir(root_dir):
                if expname in dir:
                    count += 1
                    file = pd.read_csv(os.path.join(root_dir, dir, predict_file))
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

        else:
            output_dir = 'exp_results'
            predict_file = 'predict_acc.csv'
            ensemble_file = 'ensembled_test.csv'

            file_dir = os.path.join(output_dir, f'config_{dataset}', expname, predict_file)
            file = pd.read_csv(file_dir)

            img_history = np.asarray(file['image'])
            pred_history = list(file['prediction'])

            ens_pred_history = []
            for pred in pred_history:
                pred_entry = [0.0 for _ in range(7)]
                pred_entry[int(pred)] = 1.0
                ens_pred_history.append(pred_entry)

            output_dir = os.path.join(output_dir, dataset)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            data = np.concatenate((img_history[..., np.newaxis], np.stack(ens_pred_history, axis=0)), axis=1)
            df = pd.DataFrame(data, columns=['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'])
            df.loc[len(df)] = ['ISIC_0035068', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            df.to_csv(os.path.join(output_dir, expname + '_' + ensemble_file), index=False)


def summarize_crossval_results(experiments, class_names):
    for exp in experiments:
        concat_crossval('renal', class_names, exp, topk=3)


def summarize_extval_results(experiments):
    color_dict = {'torch': 'tab:blue', 'bit-s': 'tab:orange', 'bit-m': 'tab:green'}
    results = [{exp: ens_prediction('renal', exp)} for exp in experiments]

    # plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    for result in results:
        exp = list(result.keys())[0]
        fpr, tpr, thresh, auc_score = result[exp]
        line_color = color_dict[exp.split('_')[1]]
        line_style = 'dashed' if '101' in exp else 'solid'
        plt.plot(fpr, tpr, color=line_color, lw=2, linestyle=line_style, label=f'{exp}  {auc_score}')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


def main():
    experiments = ['resnet50_torch', 'resnet50_bit-s', 'resnet101_torch']
    class_names = ['normal', 'obsolescent', 'solidified', 'disappearing', 'non-glom']
    summarize_crossval_results(experiments, class_names)

    experiments = ['resnet50_torch', 'resnet50_bit-s', 'resnet101_torch']
    summarize_extval_results(experiments)


if __name__ == '__main__':
    main()


