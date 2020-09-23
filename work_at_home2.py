from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from base.nst_base import *
from model.cnn.tcn import *
from model.cnn.resnet import *
import pathlib
import csv
import numpy as np
import os



class testnet(nn.Module):
    def __init__(self):
        super(testnet, self).__init__()
        self.conv1 = nn.Conv1d(2, 4, 200)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        # print(x.shape, x)
        x = x[:, :, -1]
        # print(x.shape, x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x


use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")

net = testnet()

if torch.cuda.device_count() > 1:
    print('use', torch.cuda.device_count(), 'GPUs')
    net = nn.DataParallel(net)

net.to(device)

transform = transforms.Compose([
    # OneHotEncoding(2),
    GoTensor(),

    # Jittering(0, 0.05),
    # TimeMask(max_width=600, use_mean=False),
    # RandomShift(600),
    # zero_pad(14463)
])
transform_val = transforms.Compose([
    # OneHotEncoding(2),
    GoTensor(),

    # zero_pad(14463)
])

base_path = pathlib.WindowsPath("C:/users/TJ_Park/Desktop/Work/Storage/3_dataset_0918_preprocessed")
train_path = pathlib.WindowsPath(base_path / "train")
test_path = pathlib.WindowsPath(base_path / "test")
valid_path = pathlib.WindowsPath(base_path / "valid")

train_dataset = NST_dataset(train_path, transform=transform, with_name=True, resolution=0.5)
valid_dataset = NST_dataset(valid_path, transform=transform_val, with_name=True, resolution=0.5)

trn_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

criterion = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)

num_epochs = 1
num_batches = len(trn_loader)

trn_loss_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
    trn_loss = 0.0
    for i, data in enumerate(trn_loader, 0):
        # print(len(trn_loader))
        inputs, labels = data['input'].to(device), data['label'].to(device)
        print(inputs.shape, inputs)
        optimizer.zero_grad()
        model_output = net(inputs)
        # print(model_output)
        # labels = labels.view(-1, 1)
        # print(model_output)
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
                    # val_label = val_label.view(-1, 1)
                    v_loss = criterion(val_output, val_label)
                    val_loss += v_loss.item()


                    # predicted = torch.round(val_output
                    predicted = torch.argmax(val_output, dim=1)
                    print(val_output)
                    print(predicted)
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

#
# csvfile = open('records/nstnet2_tcn_result_0923.csv', 'w', newline='')
# csvwriter = csv.writer(csvfile)
# for row in zip(trn_loss_list, val_loss_list, val_acc_list):
#     csvwriter.writerow(row)
# csvfile.close()