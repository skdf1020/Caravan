import pandas as pd
import pathlib
import matplotlib.pyplot as plt

df1 = pd.read_csv('nstnet_tcn_result_0923.csv', names=['trn_loss_1', 'val_loss_1', 'val_acc_1'])
df2 = pd.read_csv('nstnet2_tcn_result_0923.csv', names=['trn_loss_2', 'val_loss_2', 'val_acc_2'])
df1.plot()
df2.plot()
plt.show()