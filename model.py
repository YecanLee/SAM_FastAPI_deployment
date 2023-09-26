import torch
import torch.nn as nn
import torch.nn.functional as F

class Niconets(nn.Module):
    def __init__(self):
        super(Niconets, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p = 0.3)
        self.logsoftmax = nn.LogSoftmax(dim = 1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.logsoftmax(self.fc3(x))
        return x