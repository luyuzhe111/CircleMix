import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sn


def ensemble(dataset, expname, title=None):
    output_dir = 'exp_results'
    epoch_file = 'epoch_test.csv'
    ensemble_epoch_file = 'ensembled_epochs.csv'

    root_dir = os.path.join(output_dir, f'config_{dataset}')
    # root_dir = os.path.join(output_dir, f'config_{dataset}', 'exp1')
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

    cm = confusion_matrix(target_history,pred_history)
    print(cm)
    # df_cm = pd.DataFrame(cm, range(7), range(7))
    # sn.set(font_scale=1.4)
    # sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g', annot_kws={"size": 16}, vmin=0, vmax=2500)  # font size
    # plt.yticks(rotation=0)
    # # plt.title(title)
    # plt.savefig(os.path.join('exp_results/config_renal/confusion_matrix', title + '.svg'))
    # plt.clf()
    # # plt.show()


def main():
    print('=====Renal=====')
    # ensemble('renal', 'Efficientb0_us_aug_none', title='Baseline_h', hierarchical=False)
    # ensemble('renal', 'Efficientb0_us_aug_cutmix', title='CutMix_h', hierarchical=False)
    # ensemble('renal', 'Efficientb0_us_aug_centermix', title='CircleMix_h', hierarchical=False)

    print('=====HAM-10000=====')
    ensemble('ham', 'Efficientb0_none')
    ensemble('ham', 'Efficientb0_cutmix')
    ensemble('ham', 'Efficientb0_centermix')


if __name__ == '__main__':
    main()


