import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sn


def ensemble(dataset, expname, title=None, hierarchical=False):
    output_dir = 'exp_results'
    if hierarchical:
        # choose which child classifier to combine
        epoch_file = 'ng_combined_prediction.csv'
        ensemble_epoch_file = 'combined_ensembled_epochs_acc.csv'
    else:
        epoch_file = 'epoch_test.csv'
        ensemble_epoch_file = 'ensembled_epochs.csv'

    root_dir = os.path.join(output_dir, 'config_' + dataset)
    df = pd.DataFrame(columns=['image', 'prediction', 'target'])
    img_history = []
    pred_history = []
    target_history = []
    count = 0
    for dir in os.listdir(root_dir):
        if expname in dir and 'fold1' in dir:
            count += 1
            file = pd.read_csv(os.path.join(root_dir, dir, epoch_file))
            img_history += list(file['image'])
            if hierarchical:
                pred_history += list(file['combined_prediction'])
            else:
                pred_history += list(file['prediction'])
            target_history += list(file['target'])

    df['image'] = img_history
    df['prediction'] = pred_history
    df['target'] = target_history
    output_dir = os.path.join(output_dir, dataset)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    df.to_csv(os.path.join(output_dir, expname+'_'+ensemble_epoch_file))

    mul_accuracy = balanced_accuracy_score(target_history, pred_history)
    accuracy = accuracy_score(target_history, pred_history)
    if len(set(target_history)) == 2:
        f1 = f1_score(target_history, pred_history, average='binary')
        print(expname, 'mul accuracy: {}, f1: {}, accuracy: {}'.format(mul_accuracy, f1, accuracy))
    else:
        f1s = f1_score(target_history, pred_history, average=None)
        f1 = sum(f1s)/len(f1s)
        print(expname, 'mul accuracy: {}, f1: {}, accuracy: {}'.format(mul_accuracy, f1, accuracy))

    cm = confusion_matrix(target_history,pred_history)
    print(cm)

    if title is not None:
        df_cm = pd.DataFrame(cm, range(7), range(7))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g', annot_kws={"size": 16}, vmin=0, vmax=2500)  # font size
        plt.yticks(rotation=0)
        plt.title(title)
        plt.show()
        plt.clf()


def main():
    print('=====Renal=====')
    print('=====Plain Result=====')
    ensemble('renal', 'Efficientb0-NC2_us_aug_none', hierarchical=False)
    ensemble('renal', 'Efficientb0-NC2_us_aug_cutmix', hierarchical=False)
    ensemble('renal', 'Efficientb0-NC2_us_aug_centermix', hierarchical=False)

    ensemble('renal', 'Efficientb0-C2_us_aug_none', hierarchical=False)
    ensemble('renal', 'Efficientb0-C2_us_aug_cutmix', hierarchical=False)
    ensemble('renal', 'Efficientb0-C2_us_aug_centermix', hierarchical=False)

    ensemble('renal', 'Efficientb0-C3_us_aug_none', hierarchical=False)
    ensemble('renal', 'Efficientb0-C3_us_aug_cutmix', hierarchical=False)
    ensemble('renal', 'Efficientb0-C3_us_aug_centermix', hierarchical=False)

    ensemble('renal', 'Efficientb0-B_us_aug_none', hierarchical=False)
    ensemble('renal', 'Efficientb0-B_us_aug_cutmix', hierarchical=False)
    ensemble('renal', 'Efficientb0-B_us_aug_centermix', hierarchical=False)

    ensemble('renal', 'Efficientb0_us_aug_none', title='Baseline', hierarchical=False)
    ensemble('renal', 'Efficientb0_us_aug_cutmix', title='CutMix', hierarchical=False)
    ensemble('renal', 'Efficientb0_us_aug_centermix', title='CircleMix', hierarchical=False)

    print('=====Hierarchical Result=====')
    ensemble('renal', 'Efficientb0_us_aug_none', title='Baseline_h', hierarchical=True)
    ensemble('renal', 'Efficientb0_us_aug_cutmix', title='CutMix_h', hierarchical=True)
    ensemble('renal', 'Efficientb0_us_aug_centermix', title='CircleMix_h', hierarchical=True)


if __name__ == '__main__':
    main()


