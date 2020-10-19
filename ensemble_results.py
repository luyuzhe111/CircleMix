import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

output_dir = 'exp_results'


def ensemble(expname, best_acc):
    if best_acc:
        epoch_file = 'epoch_test_acc.csv'
        ensemble_epoch_file = 'ensembled_epochs_acc.csv'
    else:
        epoch_file = 'epoch_test.csv'
        ensemble_epoch_file = 'ensembled_epochs.csv'

    root_dir = 'exp_results/config'
    df = pd.DataFrame(columns=['image', 'prediction', 'target'])
    img_history = []
    pred_history = []
    target_history = []
    for dir in os.listdir(root_dir):
        if expname in dir:
            file = pd.read_csv(os.path.join(root_dir, dir, epoch_file))
            img_history += list(file['image'])
            pred_history += list(file['prediction'])
            target_history += list(file['target'])

    df['image'] = img_history
    df['prediction'] = pred_history
    df['target'] = target_history
    df.to_csv(os.path.join(output_dir, expname+ensemble_epoch_file))

    accuracy = accuracy_score(pred_history, target_history)
    precision = precision_score(pred_history, target_history, average='binary')
    recall = recall_score(pred_history, target_history, average='binary')
    f1 = f1_score(pred_history, target_history, average='binary')
    print(expname, 'accuracy: {}, precision: {}, recall: {}, f1: {}'.format(accuracy, precision, recall, f1))


def main():
    print('=====Select Model By Best F1=====')
    ensemble('MobileNet_os_none', False)
    ensemble('MobileNet_os_basic', False)
    ensemble('MobileNet_os_cutmix', False)
    ensemble('MobileNet_os_centermix', False)
    print()

    print('=====Select Model By Best Accuracy=====')
    ensemble('MobileNet_os_none', True)
    ensemble('MobileNet_os_basic', True)
    ensemble('MobileNet_os_cutmix', True)
    ensemble('MobileNet_os_centermix', True)


if __name__ == '__main__':
    main()


