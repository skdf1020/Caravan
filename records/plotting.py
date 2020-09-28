import pandas as pd
import pathlib
import matplotlib.pyplot as plt

df1 = pd.read_csv('nstnet_tcn_result_0923.csv', names=['trn_loss_1', 'val_loss_1', 'val_acc_1'])
df2 = pd.read_csv('nstnet2_tcn_result_0923.csv', names=['trn_loss_2', 'val_loss_2', 'val_acc_2'])
df3 = pd.read_csv('0924_nstnet_tcn1_result.csv', names=['trn_loss_3', 'val_loss_3', 'val_acc_3'])
df4 = pd.read_csv('0928_nstnet_tcn2_result.csv', names=['trn_loss_4', 'val_loss_4', 'val_acc_4'])
df5 = pd.read_csv('0928_nstnet_tcn3_result.csv', names=['trn_loss_5', 'val_loss_5', 'val_acc_5'])
# df1.plot()
# df2.plot()
# df3.plot()
df5.plot()
plt.show()