import torch
import torch.nn as nn
import torch.nn.functional as F

class IntentClassifier(nn.Module):
    def __init__(self, train_x, train_y):
        super().__init__()
        self.dense_init = nn.Linear(len(train_x[0]), 128)
        self.drop1 = nn.Dropout(0.5)
        self.dense = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.5)
        self.dense_final = nn.Linear(64, len(train_y[0]))

    def forward(self, x):
        x = F.relu(self.dense_init(x))
        x = self.drop1(x)
        x =  F.relu(self.dense(x))
        x = self.drop2(x)
        x = self.dense_final(x)
        return x