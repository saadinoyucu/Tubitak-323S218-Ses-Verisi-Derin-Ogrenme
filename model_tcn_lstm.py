"""
model_tcn_lstm.py
-----------------
TCN + LSTM hibrit modeli (PyTorch).
Orijinal model-tcn_lstm.py dosyasından kopyalanmıştır; Python import uyumluluğu
için tire yerine alt çizgi kullanılmaktadır.
"""

import torch.nn as nn
from model_tcn import TemporalBlock


class TCN_LSTM(nn.Module):
    def __init__(self, input_dim, tcn_channels=None, kernel_size=3,
                 lstm_hidden=64, dropout=0.2):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [64, 64]
        layers = []
        for i in range(len(tcn_channels)):
            in_ch         = input_dim if i == 0 else tcn_channels[i - 1]
            out_ch        = tcn_channels[i]
            dilation_size = 2 ** i
            padding       = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size,
                                        stride=1, dilation=dilation_size,
                                        padding=padding, dropout=dropout))
        self.tcn     = nn.Sequential(*layers)
        self.lstm    = nn.LSTM(input_size=tcn_channels[-1],
                               hidden_size=lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.bn      = nn.BatchNorm1d(lstm_hidden)
        self.fc1     = nn.Linear(lstm_hidden, 32)
        self.fc2     = nn.Linear(32, 1)
        self.relu    = nn.LeakyReLU(0.01)
        self.sigmoid     = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)          # (N, features, seq_len)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)          # (N, seq_len, tcn_channels[-1])
        x, (h_n, _) = self.lstm(x)
        x = self.dropout(h_n[-1])
        x = self.bn(x)                  # LSTM çıktısını stabilize et
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
