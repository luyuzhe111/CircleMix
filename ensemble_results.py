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

def concat_crossval(dataset, class_names, expname, topk=1, show_heatmap=False, verbose=False):
    topk_imgs = []
    topk_targets = []
    topk_preds = []
    topk_f1s = []
    for k in range(1, topk + 1):
        output_dir = 'exp_results'
        epoch_file = f'top{k}_epoch_test_f1.csv'
        ensemble_epoch_file = f'{expname}_top{k}_ensembled_epochs.csv'

        root_dir = os.path.join(output_dir, f'config_{dataset}')
        df = pd.DataFrame(columns=['image', 'prediction', 'target'])
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

        df['image'], df['prediction'], df['target'] = imgs, preds, targets
        topk_imgs.append(imgs)
        topk_targets.append(targets)
        topk_preds.append(preds)

        output_dir = os.path.join(output_dir, dataset)
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, ensemble_epoch_file))

        accuracy = accuracy_score(targets, preds)

        f1s = f1_score(targets, preds, average=None)
        f1 = sum(f1s)/len(f1s)
        print(expname, 'balanced f1: {}, overall accuracy: {}'.format(round(f1, 4), round(accuracy, 4)))

        cm = confusion_matrix(targets, preds)
        print(cm)

        f1s = [round(f1, 4) for f1 in f1s]
        topk_f1s.append(f1s)
        print('f1 score', f'{class_names[0]}: {f1s[0]}, {class_names[1]}: {f1s[1]}, {class_names[2]}: {f1s[2]}, {class_names[3]}: {f1s[3]}, {class_names[4]}: {f1s[4]}', '\n')

        if dataset == 'renal' and verbose:
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
            plt.savefig(os.path.join(f'exp_results/config_{dataset}/confusion_matrix', title + '.svg'))
            plt.clf()
            plt.show()

    print(f'Average results for {topk} models with 95% confidence interval')
    for i, class_name in enumerate(class_names):
        class_f1s = [topk_f1[i] for topk_f1 in topk_f1s]
        mean = sum(class_f1s) / len(class_f1s)
        std = math.sqrt(sum((x - mean) ** 2 for x in class_f1s) / len(class_f1s)) * 1.96

        print(f"{class_name} f1: {round(mean * 100, 2) } \u00B1 {round(std * 100, 2)}")


def ens_prediction(dataset, setting, expname, crossval=True):
    if dataset == 'renal':
        output_dir = 'exp_results'
        predict_file = 'predict_f1.csv'
        ensemble_file = 'ensembled_prediction.csv'

        root_dir = os.path.join(output_dir, f'config_{dataset}', setting)
        img_history = None
        pred_history = None
        target_history = None

        count = 0
        for dir in os.listdir(root_dir):
            if expname in dir:
                count += 1
                file = pd.read_csv(os.path.join(root_dir, dir, predict_file))

                if img_history is None:
                    img_history = list(file['image'])

                if target_history is None:
                    target_history = list(file['target'])

                if pred_history is None:
                    pred_history = np.asarray(list(file['pro_pos']))
                else:
                    pred_history += np.asarray(list(file['pro_pos']))

        assert count == 5, "something is wrong"

        pos_prob = list(pred_history / 5)
        df = pd.DataFrame(columns=['image', 'prob', 'target'])
        df['image'] = img_history
        df['prob'] = pos_prob
        df['target'] = target_history
        df.to_csv(os.path.join(output_dir, expname + '_' + ensemble_file))

        fpr, tpr, thresh = roc_curve(target_history, pos_prob, pos_label=1)
        auc_score = roc_auc_score(target_history, pos_prob)

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

                    if img_history is None:
                        img_history = np.asarray(file['image'])

                    pred_history.append(list(file['prediction']))

            assert count == 5, "somethig is wrong"

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
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

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


def main():
    print('=====Renal=====')
    class_names = ['normal', 'obsolescent', 'solidified', 'disappearing', 'non-glom']
    concat_crossval('renal', class_names, 'resnet50_torch', topk=3)

    # fpr1, tpr1, thresh1, auc_score1 = ens_prediction('renal', 'resnet50_randinit', 'resnet50_none')
    # fpr2, tpr2, thresh2, auc_score2 = ens_prediction('renal', 'resnet50_pytorch_pretrain', 'resnet50_none')
    # fpr3, tpr3, thresh3, auc_score3 = ens_prediction('renal', 'resnet50_bit-s', 'resnet50_none')
    # fpr4, tpr4, thresh4, auc_score4 = ens_prediction('renal', 'resnet50_bit-m', 'resnet50_none')
    # fpr5, tpr5, thresh5, auc_score5 = ens_prediction('renal', 'resnet50_ms_vision', 'resnet50_none')
    #
    # plt.plot(fpr1, tpr1, color='orange', label=f'randinit {auc_score1}')
    # plt.plot(fpr2, tpr2, color='green', label=f'pytorch pretrain  {auc_score2}')
    # plt.plot(fpr3, tpr3, color='blue', label=f'google-bit-s  {auc_score3}')
    # plt.plot(fpr4, tpr4, color='red', label=f'google-bit-m {auc_score4}')
    # plt.plot(fpr5, tpr5, color='black', label=f'microsoft {auc_score5}')
    #
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Only Glom ROC Curve')
    # plt.legend()
    # plt.show()
    # plt.close()
    #
    # fpr1, tpr1, thresh1, auc_score1 = ens_prediction('renal', 'resnet50_randinit', 'resnet50_0.25_noisy')
    # fpr2, tpr2, thresh2, auc_score2 = ens_prediction('renal', 'resnet50_pytorch_pretrain', 'resnet50_0.25_noisy')
    # fpr3, tpr3, thresh3, auc_score3 = ens_prediction('renal', 'resnet50_bit-s', 'resnet50_0.25_noisy')
    # fpr4, tpr4, thresh4, auc_score4 = ens_prediction('renal', 'resnet50_bit-m', 'resnet50_0.25_noisy')
    # fpr5, tpr5, thresh5, auc_score5 = ens_prediction('renal', 'resnet50_ms_vision', 'resnet50_0.25_noisy')
    #
    # plt.plot(fpr1, tpr1, color='orange', label=f'randinit {auc_score1}')
    # plt.plot(fpr2, tpr2, color='green', label=f'pytorch pretrain  {auc_score2}')
    # plt.plot(fpr3, tpr3, color='blue', label=f'bit-s  {auc_score3}')
    # plt.plot(fpr4, tpr4, color='red', label=f'bit-m {auc_score4}')
    # plt.plot(fpr5, tpr5, color='black', label=f'microsoft {auc_score5}')
    #
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('With Non glom ROC Curve')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()


