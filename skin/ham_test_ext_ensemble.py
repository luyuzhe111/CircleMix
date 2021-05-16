

import os
import pandas as pd


def main():
    input_dir = '//ham-10000/csv_ext_test/centermix'
    files = os.listdir(input_dir)
    files.sort()

    df_merge = pd.DataFrame()
    for pred in files:
        if 'model' in pred:
            df = pd.read_csv(os.path.join(input_dir, pred))
            df = df.rename(columns={'prediction': pred.split('.')[0]})
            if len(df_merge) == 0:
                df_merge = df.copy()
            else:
                df_merge = df_merge.merge(df, left_on='image', right_on='image', how='outer')

    maj_hist = []
    for i in range(len(df_merge)):
        preds = list(df_merge.iloc[i, 1:])
        vote_result = max(set(preds), key=preds.count)
        maj_hist.append(vote_result)

    #df_merge['final_pred'] = maj_hist


    pred_exp = []
    for i in maj_hist:
        pred_item = [0.0]*7
        pred_item[int(i)] = 1.0
        pred_exp.append(pred_item)

    df_exp = pd.DataFrame(data=pred_exp, columns=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'])
    df_exp['image'] = df_merge['image']
    df_exp.loc[len(df_exp)] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'ISIC_0035068']
    df_exp.to_csv(os.path.join(input_dir, 'centermix_ens_prediction.csv'), index=False)


if __name__ == '__main__':
    main()
