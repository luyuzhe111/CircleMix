import pandas as pd
import matplotlib.pyplot as plt


baseline = 'exp_results/config_ham/Efficientb0_none_fold1/output.csv'
cutmix = 'exp_results/config_ham/Efficientb0_cutmix_fold1/output.csv'
circlemix = 'exp_results/config_ham/Efficientb0_centermix_fold1/output.csv'

def loss_curve(df, exp):
    epochs = df['epoch_num']
    train_loss = df['train_loss']
    val_loss = df['val_loss']

    plt.plot(epochs, train_loss, label='train')
    plt.plot(epochs, val_loss, label='val')
    plt.title(exp)
    plt.legend()
    plt.show()
    plt.close()


df_bl = pd.read_csv(baseline, index_col=0)
df_cutmix = pd.read_csv(cutmix, index_col=0)
df_circlemix = pd.read_csv(circlemix, index_col=0)

loss_curve(df_bl, 'baseline')
loss_curve(df_cutmix, 'cutmix')
loss_curve(df_circlemix, 'circlemix')
