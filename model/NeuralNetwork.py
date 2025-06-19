import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.up_projection = nn.Linear(model_dim, model_dim*4)
        self.relu = nn.ReLU()
        self.down_projection = nn.Linear(model_dim*4, model_dim)
        self.dropout = nn.Dropout(0.2) # using p = 0.2

    def forward(self, x):
        torch.manual_seed(0)
        return self.dropout(self.down_projection(self.relu(self.up_projection(x))))