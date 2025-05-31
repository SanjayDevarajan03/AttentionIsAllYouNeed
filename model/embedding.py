import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size,d_model,max_len=500):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        # Positional encoding matrix (1, max_len, d_model) 
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        self.token_embedding[:, 0::2] = torch.sin(position*div_term)
        self.token_embedding[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        token_emb = self.token_embedding(x) * math.sqrt(self.d_model)
        pos_enc = self.pe[:, :seq_len, :].to(x.device)
        return token_emb + pos_enc # Shape: (batch_size, seq_len, d_model)
