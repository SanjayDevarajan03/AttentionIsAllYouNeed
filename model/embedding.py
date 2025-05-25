import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, Tuple


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.Embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.Embedding(input_ids)
    

class PositionalEmbedding:
    def __init__(self, vocab_size, d_model, max_len = 500):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        # Positional encoding matrix (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model,2 ).float() * (-math.log(10000.0) / d_model))

        self.token_embedding[:, 0::2] = torch.sin(position * div_term)
        self.token_embedding[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        token_emb = self.token_embedding(x) * math.sqrt(self.d_model)
        pos_enc = self.pe[:, :seq_len, :].to(x.device)
        return token_emb + pos_enc # Shape: (batch_size, seq_len, d_model)

'''
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embedding_dim):
        super().__init__()
        self.encoding = torch.zeros(max_len, embedding_dim)
        self.encoding.requires_grad = False

        position = torch.arange(0, max_len, dtype=torch.float32)
        position = position.unsqueeze(1)

        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / embedding_dim))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)

        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, : x.size(1)]
    

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim)

    def forward(self, x):
        return self.embedding + math.sqrt(self.embedding_dim)
'''