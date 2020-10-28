from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from NST.base import *
from model.cnn.tcn import *
from model.cnn.mobilenet import *
from model.cnn.mobilenetV2 import *
import pathlib
import csv

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# net = TCN(2, 6, [16, 16, 64, 64, 256, 256, 512, 512], 3, 0)
# net = MobileNetV1_1d(ch_in=2, n_classes=2)
net = MobileNetV2()

if torch.cuda.device_count() > 1:
    print('use', torch.cuda.device_count(), 'GPUs')
    net = nn.DataParallel(net)

net.to(device)

transform = transforms.Compose([
    GoTensor(),
    Jittering(0, 0.05),
    Permutation(5),
    TimeMask(max_width=1000, use_mean=False, is_1d=True),
    # # RandomShift(0),
    # zero_pad(527)
])
transform_val = transforms.Compose([
    GoTensor(),
    # zero_pad(527)
])

base_path = pathlib.PosixPath("/home/tj/Torch/Data/Datafolder/3_dataset_1007_preprocessed")
train_path = pathlib.PosixPath(base_path / "train")
test_path = pathlib.PosixPath(base_path / "test")
valid_path = pathlib.PosixPath(base_path / "valid")

train_dataset = NST_dataset(train_path, transform=transform, have_name=True, resolution=1)
valid_dataset = NST_dataset(valid_path, transform=transform_val, have_name=True, resolution=1)

trn_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)


# weights = torch.tensor([478.0, 71.0, 20.0, 337.0, 21., 17.0], dtype=torch.float32)
# weights = weights / weights.sum()
# print(weights)
# weights = 1.0 / weights
# weights = weights / weights.sum()
# print(weights)
# criterion = nn.CrossEntropyLoss(weight=weights.cuda())
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = optim.Adam(net.parameters(), lr=learning_rate)#, weight_decay=1e-4)

num_epochs = 200
num_batches = len(trn_loader)

trn_loss_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
    trn_loss = 0.0
    for i, data in enumerate(trn_loader, 0):
        inputs, labels = data['input'].to(device), data['label'].to(device)

        optimizer.zero_grad()
        model_output = net(inputs)
        print(model_output)

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
                    v_loss = criterion(val_output, val_label)
                    val_loss += v_loss.item()

                    predicted = torch.argmax(val_output, dim=1)
                    # print(val_output)
                    print(predicted)
                    print(val_label)
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


csvfile = open('/home/tj/PycharmProjects/Caravan/records/1028_nstnet_mb9_result.csv', 'w', newline='')
csvwriter = csv.writer(csvfile)
for row in zip(trn_loss_list, val_loss_list, val_acc_list):
    csvwriter.writerow(row)
csvfile.close()
