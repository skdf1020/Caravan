import torch
import torch.nn as nn
import torch.nn.functional as F


# t4d = torch.empty(1, 1, 4)
# p1d = (10, 10)
#
# print(F.pad(t4d, p1d, 'constant', 0))

class testnet(nn.Module):
    def __init__(self):
        super(testnet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=1, dilation=1, groups=4, bias=False)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=8, out_features=2)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        print('after conv ', x, x.shape)
        # x = self.relu(x)
        x = x[:, :, -1]
        print('after chomp ', x, x.shape)
        x = self.fc(x)
        print('output is ', x, x.shape)
        # x = self.relu(x)
        # x = self.softmax(x)
        return x

loss = nn.CrossEntropyLoss()
input = torch.randn(1, 4, 5, requires_grad=True)
target = torch.empty(1, dtype=torch.long).random_(2)
print('input is ', input)
print('target is ', target)
net = testnet()
print(list(net.named_parameters()))
output = net(input)
loss = loss(output, target).item()
print('loss is ', loss)