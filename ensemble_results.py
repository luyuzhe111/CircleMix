import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sn


def concat_crossval(dataset, expname):
    output_dir = 'exp_results'
    epoch_file = 'top3_epoch_test_f1.csv'
    ensemble_epoch_file = 'ensembled_epochs.csv'

    root_dir = os.path.join(output_dir, f'config_{dataset}')
    df = pd.DataFrame(columns=['image', 'prediction', 'target'])
    img_history = []
    pred_history = []
    target_history = []
    count = 0
    for dir in os.listdir(root_dir):
        if expname in dir:
            count += 1
            file = pd.read_csv(os.path.join(root_dir, dir, epoch_file))
            img_history += list(file['image'])
            pred_history += list(file['prediction'])
            target_history += list(file['target'])

            # if dataset == 'renal':
            #     pred_history = list(map(lambda x: 0 if x == 4 else x, pred_history))
            #     target_history = list(map(lambda x: 0 if x == 4 else x, target_history))

    assert count == 5, "somethig is wrong"

    df['image'] = img_history
    df['prediction'] = pred_history
    df['target'] = target_history
    output_dir = os.path.join(output_dir, dataset)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    df.to_csv(os.path.join(output_dir, expname+'_'+ensemble_epoch_file))

    mul_accuracy = balanced_accuracy_score(target_history, pred_history)
    accuracy = accuracy_score(target_history, pred_history)

    f1s = f1_score(target_history, pred_history, average=None)
    f1 = sum(f1s)/len(f1s)
    print(expname, 'mul accuracy: {}, f1: {}, accuracy: {}'.format(mul_accuracy, f1, accuracy))

    cm = confusion_matrix(target_history, pred_history)
    print(cm)
    print('acc', list(cm.diagonal() / cm.sum(axis=1)))
    print('f1', f1s, '\n')
    #
    # if dataset == 'renal':
    #     bi_tar_history = list(map(lambda x: 0 if x == 0 else 1, target_history))
    #     bi_pred_history = list(map(lambda x: 0 if x == 0 else 1, pred_history))
    #
    #     bi_cm = confusion_matrix(bi_tar_history, bi_pred_history)
    #     print(expname, 'accuracy: {}, f1: {}'.format(accuracy_score(bi_tar_history, bi_pred_history), f1_score(bi_tar_history, bi_pred_history), '\n'))
    #     print(bi_cm)

    # df_cm = pd.DataFrame(cm, range(7), range(7))
    # sn.set(font_scale=1.4)
    # sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g', annot_kws={"size": 16}, vmin=0, vmax=2500)  # font size
    # plt.yticks(rotation=0)
    # plt.title(title)
    # plt.savefig(os.path.join('exp_results/config_renal/confusion_matrix', title + '.svg'))
    # plt.clf()
    # plt.show()


def ens_prediction(dataset, setting, expname, crossval=True):
    if dataset == 'renal':
        output_dir = 'exp_results'
        predict_file = 'predict_f1.csv'
        ensemble_file = 'ensembled_prediction.csv'

        root_dir = os.path.join(output_dir, f'config_{dataset}', 'without_fibrosis', setting)
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

        assert count == 5, "somethig is wrong"

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
    # concat_crossval('renal', 'resnet50_none')
    # concat_crossval('renal', 'resnet50_0.25_noisy')
    # concat_crossval('renal', 'efficientb0_none')
    # concat_crossval('renal', 'efficientb0_cutout')
    # concat_crossval('renal', 'efficientb0_cutmix')
    # concat_crossval('renal', 'efficientb0_circlemix')

    fpr1, tpr1, thresh1, auc_score1 = ens_prediction('renal', 'resnet50_randinit', 'resnet50_none')
    fpr2, tpr2, thresh2, auc_score2 = ens_prediction('renal', 'resnet50_pytorch_pretrain', 'resnet50_none')
    fpr3, tpr3, thresh3, auc_score3 = ens_prediction('renal', 'resnet50_bit-s', 'resnet50_none')
    fpr4, tpr4, thresh4, auc_score4 = ens_prediction('renal', 'resnet50_bit-m', 'resnet50_none')
    fpr5, tpr5, thresh5, auc_score5 = ens_prediction('renal', 'resnet50_ms_vision', 'resnet50_none')

    plt.plot(fpr1, tpr1, color='orange', label=f'randinit {auc_score1}')
    plt.plot(fpr2, tpr2, color='green', label=f'pytorch pretrain  {auc_score2}')
    plt.plot(fpr3, tpr3, color='blue', label=f'google-bit-s  {auc_score3}')
    plt.plot(fpr4, tpr4, color='red', label=f'google-bit-m {auc_score4}')
    plt.plot(fpr5, tpr5, color='black', label=f'microsoft {auc_score5}')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Only Glom ROC Curve')
    plt.legend()
    plt.show()
    plt.close()

    fpr1, tpr1, thresh1, auc_score1 = ens_prediction('renal', 'resnet50_randinit', 'resnet50_0.25_noisy')
    fpr2, tpr2, thresh2, auc_score2 = ens_prediction('renal', 'resnet50_pytorch_pretrain', 'resnet50_0.25_noisy')
    fpr3, tpr3, thresh3, auc_score3 = ens_prediction('renal', 'resnet50_bit-s', 'resnet50_0.25_noisy')
    fpr4, tpr4, thresh4, auc_score4 = ens_prediction('renal', 'resnet50_bit-m', 'resnet50_0.25_noisy')
    fpr5, tpr5, thresh5, auc_score5 = ens_prediction('renal', 'resnet50_ms_vision', 'resnet50_0.25_noisy')

    plt.plot(fpr1, tpr1, color='orange', label=f'randinit {auc_score1}')
    plt.plot(fpr2, tpr2, color='green', label=f'pytorch pretrain  {auc_score2}')
    plt.plot(fpr3, tpr3, color='blue', label=f'bit-s  {auc_score3}')
    plt.plot(fpr4, tpr4, color='red', label=f'bit-m {auc_score4}')
    plt.plot(fpr5, tpr5, color='black', label=f'microsoft {auc_score5}')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('With Non glom ROC Curve')
    plt.legend()
    plt.show()

    # print('=====HAM-10000=====')
    # ens_prediction('ham', 'efficientb0_none')
    # ens_prediction('ham', 'efficientb0_cutmix')
    # ens_prediction('ham', 'efficientb0_circlemix')
    #
    # concat_crossval('ham', 'efficientb0_none')
    # concat_crossval('ham', 'efficientb0_cutmix')
    # concat_crossval('ham', 'efficientb0_circlemix')

    # ens_prediction('ham', 'efficientb0_cutmix_train_val', crossval=False)
    # ens_prediction('ham', 'efficientb0_circlemix_train_val', crossval=False)


if __name__ == '__main__':
    main()


