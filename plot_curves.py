import pandas as pd
import matplotlib.pyplot as plt


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


df1 = pd.read_csv('exp_results/config_renal/seed5/output.csv', index_col=0)
df2 = pd.read_csv('exp_results/config_renal/seed10/output.csv', index_col=0)

num_cls = 5
fig, axes = plt.subplots(nrows=1, ncols=num_cls, figsize=(15, 3))
labels = ['normal', 'obsolescent', 'solidified', 'disappearing', 'fibrosis']
for idx, label in enumerate(labels):
    x1 = df1['epoch_num']
    x2 = df2['epoch_num']

    y1 = df1[label]
    y2 = df2[label]

    axes[idx].plot(x1, y1, label='seed5')
    axes[idx].plot(x2, y2, label='seed10')
    axes[idx].set_title(label)

fig.tight_layout()
plt.legend()
plt.show()