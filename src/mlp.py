import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, X):
        return self.net(X)
