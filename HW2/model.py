import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256, dropout_rate=0.5):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            # BasicBlock(input_dim, hidden_dim, dropout_rate),
            # *[BasicBlock(hidden_dim, hidden_dim, dropout_rate) for _ in range(hidden_layers)],
            # nn.Linear(hidden_dim, output_dim)
            nn.Linear(input_dim, 2048), # 1
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024), # 2
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512), # 2
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256), # 3
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128), # 4
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class BiRNN(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256, dropout_rate=0.5):
        super(BiRNN, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, hidden_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)  #  *2 for bidirectional

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # initialize hidden and cell state
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0)) # through RNN
        out = self.fc(out[:, -1, :]) # final dense layer
        return out