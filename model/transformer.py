import torch
import torch.nn as nn
from torchtyping import TensorType
from attention import MultiHeadAttention
from NeuralNetwork import NeuralNetwork



class Transformer(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.attention = MultiHeadAttention(model_dim, num_heads)
        self.linear_network = NeuralNetwork(model_dim)
        self.first_norm = nn.LayerNorm(model_dim)
        self.second_norm = nn.LayerNorm(model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        embedded = embedded + self.attention(self.first_norm(embedded)) # skip connection
        embedded = embedded + self.linear_network(self.second_norm(embedded))
        return torch.round(embedded(decimals=4))




    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)