import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
<<<<<<< HEAD:model/cnn/tcn2.py
        # print(chomp_size)
    # def forward(self, x):
    #     return x[:, :, self.chomp_size:].contiguous()
    def forward(self, x):
        # print(x)
        x = x[:, :, :-self.chomp_size].contiguous()
        # print(x)
=======

    def forward(self, x):
        x = x[:, :, :-self.chomp_size].contiguous()
>>>>>>> dcf8988239eac7aa8b8b71195ba9eedc7c6dc1e5:model/cnn/tcn.py
        return x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(n_inputs)
        # self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
        #                                    stride=stride, padding=padding, dilation=dilation))
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout)

        self.bn2 = nn.BatchNorm1d(n_outputs)
        # self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
        #                                    stride=stride, padding=(kernel_size-1)*1, dilation=1))
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=(kernel_size - 1) * 1, dilation=1)
        self.chomp2 = Chomp1d((kernel_size-1)*1)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout)

        self.net = nn.Sequential(self.bn1, self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.bn2, self.conv2, self.chomp2, self.relu2, self.dropout2)
        # self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
        #                          self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
<<<<<<< HEAD:model/cnn/tcn2.py
        # print(x)
        res = x if self.downsample is None else self.downsample(x)
        x = self.conv1(x)
        # print(x)
        x = self.chomp1(x)
        # print(x)
        x = self.relu1(x)
        # print(x)
        x = self.dropout1(x)
        # print(x)
        out = x
        # out = self.net(x)

        # res = x if self.downsample is None else self.downsample(x)
        # print(res)
        # print(out+res)
=======
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class TemporalBlock2(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock2, self).__init__()
        self.bn1 = nn.BatchNorm1d(n_inputs)
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()

        self.bn_b1 = nn.BatchNorm1d(n_outputs)
        self.bottleneck1 = nn.Conv1d(n_outputs, n_outputs*3, kernel_size, padding=5)
        self.relu_b1 = nn.ReLU()

        self.bn_b2 = nn.BatchNorm1d(n_outputs*3)
        self.bottleneck2 = nn.Conv1d(n_outputs*3, n_outputs, kernel_size, padding=5)
        self.relu_b2 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout)

        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=(kernel_size - 1) * 1, dilation=1)
        self.chomp2 = Chomp1d((kernel_size-1)*1)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout)

        self.net = nn.Sequential(self.bn1, self.conv1, self.chomp1, self.relu1,
                                 self.bn_b1, self.bottleneck1, self.relu_b1,
                                 self.bn_b2, self.bottleneck2, self.relu_b2, self.dropout1,
                                 self.bn2, self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        self.bottleneck1.weight.data.normal_(0, 0.01)
        self.bottleneck2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
>>>>>>> dcf8988239eac7aa8b8b71195ba9eedc7c6dc1e5:model/cnn/tcn.py

        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=16, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock2(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.linear.weight.data.normal_(0, 0.01)
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        # print(y1)
        # print('last', y1[:, :, -1])
        # print('last', y1[:, :, -2])
        # print('last', y1[:, :, -3])
        # print('last', y1[:, :, -4])
        # print('last', y1[:, :, -5])
        # print('last', y1[:, :, 1])
        # print('last', y1[:, :, 2])
        # print('last', y1[:, :, 3])
        # print('last', y1[:, :, 4])
        # print('last', y1[:, :, 5])

        o = self.linear(y1[:, :, -1])
        # result = F.log_softmax(o, dim=1)
        # print(result, result.shape)
        o = self.sigmoid(o)
        # print('out', o)
        # print(o.shape)
        return o

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features