# from torchvision import transforms
# from torch.utils.data import DataLoader
# import torch.optim as optim
# from NST.base import *
# from model.cnn.tcn import *
# import pathlib
# import csv
#
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
#
# # net = TCN(2, 6, [16, 16, 52, 52, 256, 256, 512, 512], 3, 0.1)
# net = TCN(2, 6, [4, 4, 8, 8], 601, 0.1)
#
# if torch.cuda.device_count() > 1:
#     print('use', torch.cuda.device_count(), 'GPUs')
#     net = nn.DataParallel(net)
#
# net.to(device)
#
# transform = transforms.Compose([
#     GoTensor(),
#     Jittering(0, 0.05),
#     TimeMask(max_width=500, use_mean=False, is_1d=True),
#     RandomShift(500),
#     zero_pad(11401)
# ])
# transform_val = transforms.Compose([
#     GoTensor(),
#     zero_pad(11401)
# ])
#
# base_path = pathlib.WindowsPath("C:/users/TJ_Park/Desktop/Work/Storage/3_dataset_0929_preprocessed")
# train_path = pathlib.WindowsPath(base_path / "train")
# test_path = pathlib.WindowsPath(base_path / "test")
# valid_path = pathlib.WindowsPath(base_path / "valid")
#
# train_dataset = NST_dataset(train_path, transform=transform, have_name=True, resolution=1)
# valid_dataset = NST_dataset(valid_path, transform=transform_val, have_name=True, resolution=1)
#
# trn_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# val_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
#
#
# weights = torch.tensor([478.0, 71.0, 20.0, 337.0, 21., 17.0], dtype=torch.float32)
# weights = weights / weights.sum()
# print(weights)
# weights = 1.0 / weights
# weights = weights / weights.sum()
# print(weights)
# criterion = nn.CrossEntropyLoss(weight=weights.cuda())
# learning_rate = 1e-3
# optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
#
# num_epochs = 30
# num_batches = len(trn_loader)
#
# trn_loss_list = []
# val_loss_list = []
# val_acc_list = []
#
# for epoch in range(num_epochs):
#     trn_loss = 0.0
#     for i, data in enumerate(trn_loader, 0):
#         inputs, labels = data['input'].to(device), data['label'].to(device)
#         print(inputs.shape, labels.shape)
#         optimizer.zero_grad()
#         model_output = net(inputs)
#
#         loss = criterion(model_output, labels)
#
#         # name = data['name']
#         # print(name)
#
#         loss.backward()
#         optimizer.step()
#
#         trn_loss += loss.item()
#
#         # for param in net.parameters():
#         #     print(param.data)
#         if i % 5 == 4:
#             with torch.no_grad():
#                 val_loss = 0.0
#                 total, correct = 0, 0
#                 for j, val in enumerate(val_loader, 0):
#                     val_x, val_label = val['input'].to(device), val['label'].to(device)
#
#                     val_output = net(val_x)
#                     v_loss = criterion(val_output, val_label)
#                     val_loss += v_loss.item()
#
#                     predicted = torch.argmax(val_output, dim=1)
#                     # print(val_output)
#                     print(predicted)
#                     print(val_label)
#                     total += val_label.size(0)
#                     correct += (predicted == val_label).sum().item()
#
#                 print("epoch: {}/{} | trn loss: {:.4f} | val loss: {:.4f} | val acc: {:.4f}".format(
#                     epoch + 1,
#                     num_epochs,
#                     trn_loss / 5,
#                     val_loss / len(val_loader),
#                     correct / total
#                 ))
#                 print(correct, total)
#
#                 trn_loss_list.append(trn_loss / 5)
#                 val_loss_list.append(val_loss / len(val_loader))
#                 val_acc_list.append(correct / total)
#             trn_loss = 0.0
#
# print('Finished Training')
#
#
# csvfile = open('C:/users/TJ_Park/PycharmProjects/homework/records/1002_nstnet_tcn5_result.csv', 'w', newline='')
# csvwriter = csv.writer(csvfile)
# for row in zip(trn_loss_list, val_loss_list, val_acc_list):
#     csvwriter.writerow(row)
# csvfile.close()


from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from NST.base import *
from model.cnn.tcn import *
import pathlib
import csv

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# net = TCN(2, 6, [16, 16, 52, 52, 256, 256, 512, 512], 3, 0.1)
net = TCN(2, 1, [4, 4, 8, 8], 601, 0)

if torch.cuda.device_count() > 1:
    print('use', torch.cuda.device_count(), 'GPUs')
    net = nn.DataParallel(net)

net.to(device)

transform = transforms.Compose([
    GoTensor(),
    Jittering(0, 0.05),
    TimeMask(max_width=500, use_mean=False, is_1d=True),
    RandomShift(500),
    zero_pad(11401)
])
transform_val = transforms.Compose([
    GoTensor(),
    zero_pad(11401)
])

base_path = pathlib.WindowsPath("C:/users/TJ_Park/Desktop/Work/Storage/3_dataset_1002_preprocessed")
train_path = pathlib.WindowsPath(base_path / "train")
test_path = pathlib.WindowsPath(base_path / "test")
valid_path = pathlib.WindowsPath(base_path / "valid")

train_dataset = NST_dataset(train_path, transform=transform, have_name=True, resolution=1)
valid_dataset = NST_dataset(valid_path, transform=transform_val, have_name=True, resolution=1)

trn_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)


# weights = torch.tensor([478.0, 71.0, 20.0, 337.0, 21., 17.0], dtype=torch.float32)
# weights = weights / weights.sum()
# print(weights)
# weights = 1.0 / weights
# weights = weights / weights.sum()
# print(weights)
# criterion = nn.CrossEntropyLoss(weight=weights.cuda())
criterion = nn.BCELoss()
learning_rate = 1e-2
optimizer = optim.Adam(net.parameters(), lr=learning_rate)#, weight_decay=1e-5)

num_epochs = 20
num_batches = len(trn_loader)

trn_loss_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
    trn_loss = 0.0
    for i, data in enumerate(trn_loader, 0):
        inputs, labels = data['input'].to(device), data['label'].to(device)
        print(inputs.shape, labels.shape, type(labels))
        optimizer.zero_grad()
        model_output = net(inputs)
        labels = labels.view(-1, 1)
        loss = criterion(model_output, labels)

        # name = data['name']
        # print(name)

        loss.backward()
        optimizer.step()

        trn_loss += loss.item()

        # for param in net.parameters():
        #     print(param.data)
        if i % 5 == 4:
            with torch.no_grad():
                val_loss = 0.0
                total, correct = 0, 0
                for j, val in enumerate(val_loader, 0):
                    val_x, val_label = val['input'].to(device), val['label'].to(device)

                    val_output = net(val_x)
                    val_label = val_label.view(-1, 1)
                    v_loss = criterion(val_output, val_label)
                    val_loss += v_loss.item()

                    predicted = torch.round(val_output)
                    total += val_label.size(0)
                    correct += (predicted == val_label).sum().item()

                print("epoch: {}/{} | trn loss: {:.4f} | val loss: {:.4f} | val acc: {:.4f}".format(
                    epoch + 1,
                    num_epochs,
                    trn_loss / 5,
                    val_loss / len(val_loader),
                    correct / total
                ))
                print(correct, total)

                trn_loss_list.append(trn_loss / 5)
                val_loss_list.append(val_loss / len(val_loader))
                val_acc_list.append(correct / total)
            trn_loss = 0.0

print('Finished Training')


csvfile = open('C:/users/TJ_Park/PycharmProjects/homework/records/1004_nstnet_tcn9_result.csv', 'w', newline='')
csvwriter = csv.writer(csvfile)
for row in zip(trn_loss_list, val_loss_list, val_acc_list):
    csvwriter.writerow(row)
csvfile.close()