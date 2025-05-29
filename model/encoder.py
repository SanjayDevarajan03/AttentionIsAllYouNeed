import torch
import torch.nn as nn
from embedding import PositionalEmbedding

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_num, max_len, dropout):
        super().__init__()

        self.embedding = PositionalEmbedding(vocab_size, embedding_dim)

    def forward(self, x):
        return 
