"""
model_tcn.py
------------
TCN (Temporal Convolutional Network) modeli (PyTorch).
Orijinal model-tcn.py dosyasından kopyalanmıştır; Python import uyumluluğu için
tire yerine alt çizgi kullanılmaktadır.
"""

import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


from torch.nn.utils import weight_norm

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride,
                 dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1    = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                              stride=stride, padding=padding, dilation=dilation))
        self.chomp1   = Chomp1d(padding)
        self.relu1    = nn.LeakyReLU(0.01)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2    = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                              stride=stride, padding=padding, dilation=dilation))
        self.chomp2   = Chomp1d(padding)
        self.relu2    = nn.LeakyReLU(0.01)
        self.dropout2 = nn.Dropout(dropout)
        
        self.net      = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = (weight_norm(nn.Conv1d(n_inputs, n_outputs, 1))
                           if n_inputs != n_outputs else None)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, input_dim, num_channels=None, kernel_size=3, dropout=0.2):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 128, 64]
        layers = []
        for i in range(len(num_channels)):
            in_ch         = input_dim if i == 0 else num_channels[i - 1]
            out_ch        = num_channels[i]
            dilation_size = 2 ** i
            padding       = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size,
                                        stride=1, dilation=dilation_size,
                                        padding=padding, dropout=dropout))
        self.tcn         = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.bn          = nn.BatchNorm1d(num_channels[-1])
        self.fc1         = nn.Linear(num_channels[-1], 32)
        self.fc2         = nn.Linear(32, 1)
        self.relu        = nn.LeakyReLU(0.01)
        self.sigmoid     = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)          # (N, features, seq_len)
        x = self.tcn(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.bn(x)                  # Batch norm ile patlamaları engelle
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
