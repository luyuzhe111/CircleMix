import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('csv/ground_truth.csv')

sum_series = df.sum(axis=0, skipna=True)
sum_dict = dict(sum_series.items())
sum_dict.pop('image', None)

x = list(sum_dict.keys())
y = list(sum_dict.values())
plt.barh(x, y)

for index, value in enumerate(y):
    plt.text(value, index, str(value).split('.')[0])

plt.show()

print('There are {} images in total'.format(sum(y)))
