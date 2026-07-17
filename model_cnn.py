"""
model_cnn.py
------------
CNN + LSTM hibrit modeli (PyTorch).
Orijinal model-cnn.py dosyasından kopyalanmıştır; Python import uyumluluğu için
tire yerine alt çizgi kullanılmaktadır.

NOT: MFCC özellikleri 40-dim sabit vektörlerdir → girdi şekli (N, 1, 40).
     MaxPool1d(2) seq_len=1 üzerinde çalışamaz; bu yüzden AdaptiveMaxPool1d(1)
     kullanılmaktadır — her seq_len değerinde güvenle çalışır.
"""

import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Conv1d: in_channels=input_dim (40 for MFCC), seq_len üzerinde kernel uygular
        self.conv1   = nn.Conv1d(in_channels=input_dim, out_channels=64,
                                 kernel_size=3, padding=1)
        # AdaptiveMaxPool: seq_len ne olursa olsun çıkışı 1'e indirir
        self.pool1   = nn.AdaptiveMaxPool1d(1)
        self.conv2   = nn.Conv1d(64, 64, kernel_size=1)   # kernel=1: seq_len=1 için güvenli
        self.pool2   = nn.AdaptiveMaxPool1d(1)
        self.lstm    = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1     = nn.Linear(64, 32)
        self.fc2     = nn.Linear(32, 1)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (N, seq_len, input_dim)  →  (N, 1, 40) için MFCC özellikleri
        x = x.permute(0, 2, 1)              # → (N, input_dim, seq_len) = (N, 40, 1)
        x = self.relu(self.conv1(x))        # → (N, 64, 1)
        x = self.pool1(x)                   # → (N, 64, 1)
        x = self.relu(self.conv2(x))        # → (N, 64, 1)
        x = self.pool2(x)                   # → (N, 64, 1)
        x = x.permute(0, 2, 1)              # → (N, 1, 64)
        x, (h_n, _) = self.lstm(x)
        x = self.dropout(h_n[-1])           # Son hidden state: (N, 64)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
