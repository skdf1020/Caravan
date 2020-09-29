from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from NST.base import *
from model.cnn.tcn import *
import pathlib
import csv


use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
# net = TCN(2, 1, [16, 16, 16, 16, 16, 16], 151, 0.2)
net = TCN(2, 1, [16, 16, 16, 16, 16, 16, 16, 16, 16, 16], 15, 0.2)

if torch.cuda.device_count() > 1:
    print('use', torch.cuda.device_count(), 'GPUs')
    net = nn.DataParallel(net)

net.to(device)

transform = transforms.Compose([
    GoTensor(),
    Jittering(0, 0.05),
    TimeMask(max_width=600, use_mean=False),
    RandomShift(600),
    zero_pad(14463)
])
transform_val = transforms.Compose([
    GoTensor(),
    zero_pad(14463)
])

base_path = pathlib.WindowsPath("C:/users/TJ_Park/Desktop/Work/Storage/3_dataset_0918_preprocessed")
train_path = pathlib.WindowsPath(base_path / "train")
test_path = pathlib.WindowsPath(base_path / "test")
valid_path = pathlib.WindowsPath(base_path / "valid")

train_dataset = NST_dataset(train_path, transform=transform, with_name=True)
valid_dataset = NST_dataset(valid_path, transform=transform_val, with_name=True)

trn_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

criterion = nn.BCELoss()
learning_rate = 1e-3
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)

num_epochs = 100
num_batches = len(trn_loader)

trn_loss_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
    trn_loss = 0.0
    for i, data in enumerate(trn_loader, 0):
        # print(len(trn_loader))
        inputs, labels = data['input'].to(device), data['label'].to(device)
        # print(inputs.shape)
        optimizer.zero_grad()
        model_output = net(inputs)
        labels = labels.view(-1, 1)
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


csvfile = open('../records/nstnet2_tcn_result_0923.csv', 'w', newline='')
csvwriter = csv.writer(csvfile)
for row in zip(trn_loss_list, val_loss_list, val_acc_list):
    csvwriter.writerow(row)
csvfile.close()