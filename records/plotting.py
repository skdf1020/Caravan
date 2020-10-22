import pandas as pd
import pathlib
import matplotlib.pyplot as plt


# df0_0 = pd.read_csv('nstnet_tcn_result_0921.csv', names=['trn_loss_0_0', 'val_loss_0_0', 'val_acc_0_0'])
# df0_1 = pd.read_csv('nstnet_tcn_result_0923.csv', names=['trn_loss_0_1', 'val_loss_0_1', 'val_acc_0_1'])
# df0_2 = pd.read_csv('nstnet2_tcn_result_0923.csv', names=['trn_loss_0_2', 'val_loss_0_2', 'val_acc_0_2'])
#
# df1_0 = pd.read_csv('0924_nstnet_tcn1_result.csv', names=['trn_loss_1_0', 'val_loss_1_0', 'val_acc_1_0'])
# df1_1 = pd.read_csv('0928_nstnet_tcn2_result.csv', names=['trn_loss_1_1', 'val_loss_1_1', 'val_acc_1_1'])
# df1_2 = pd.read_csv('0928_nstnet_tcn3_result.csv', names=['trn_loss_1_2', 'val_loss_1_2', 'val_acc_1_2'])
#
# df1 = pd.read_csv('1002_nstnet_tcn4_result.csv', names=['trn_loss_1', 'val_loss_1', 'val_acc_1'])
# df2 = pd.read_csv('1002_nstnet_tcn5_result.csv', names=['trn_loss_2', 'val_loss_2', 'val_acc_2'])
# df3 = pd.read_csv('1002_nstnet_tcn6_result.csv', names=['trn_loss_3', 'val_loss_3', 'val_acc_3'])
# df4 = pd.read_csv('1002_nstnet_tcn7_result.csv', names=['trn_loss_4', 'val_loss_4', 'val_acc_4'])
# df5 = pd.read_csv('1002_nstnet_tcn8_result.csv', names=['trn_loss_5', 'val_loss_5', 'val_acc_5'])
#
#
# df0_0.plot()
# df0_1.plot()
# df0_2.plot()
#
# df1_0.plot()
# df1_1.plot()
# df1_2.plot()
#
# df1.plot()
# df2.plot()
# df3.plot()
# df4.plot()
# df5.plot()
# plt.show()

df= pd.read_csv('1004_nstnet_tcn9_result.csv', names=['trnLoss','valLoss','valACC'])
df.plot()
plt.show()