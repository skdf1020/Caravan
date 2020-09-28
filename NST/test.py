# import torch
# import torch.nn.functional as F
#
# t4d = torch.empty(3, 3, 4, 2)
# p1d = (0,10)
#
# print(F.pad(t4d, p1d, 'constant', 0))
#
# class testnet(nn.Module):
#     def __init__(self):
#         super(testnet, self).__init__()
#         self.conv1 = nn.Conv1d(2, 4, 200)
#         self.relu = nn.ReLU()
#         self.fc = nn.Linear(4, 2)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         # print(x.shape, x)
#         x = x[:, :, -1]
#         # print(x.shape, x)
#         x = self.fc(x)
#         x = self.relu(x)
#         x = self.softmax(x)
#         return x

def recep_cal(kernel, layer):
    result = 1
    for i in range(layer):
        result += (kernel-1)*(2**i+1)
    return result

print(recep_cal(3, 8))

# import torch, torch.nn as nn
#
# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# print(input, target)
# output = loss(input, target)
# print(output)
