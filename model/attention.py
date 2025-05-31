import torch
import torch.nn as nn
import numpy as np

class ScaledAttention(nn.Module):
    def __init__(self, query, key, value, vocab_size, d_model):
        super().__init__()
        query = nn.Embedding(vocab_size, d_model)
        key = nn.Embedding(vocab_size, d_model)
        value = nn.Embedding(vocab_size, d_model)

        attention = nn.Softmax((query*torch.transpose(key))/np.sqrt(len(key)))* value



class MultiHeadAttention:
    def __init__(self, query, key, value, vocab_size):
        super().__init__()
        self.query_embedding = nn.Embedding(vocab_size, len(key))
        self.key_embedding = nn.Embedding(vocab_size, len(key))
        self.value_embedding = nn.Embedding(vocab_size, len(value))

    def forward(self):
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU()
        )


    











