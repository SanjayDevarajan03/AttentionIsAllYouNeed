import torch
import torch.nn as nn

class ScaledAttention(nn.Module):
    def __init__(self, query, key, value, vocab_size):
        super().__init__()
        query = nn.Embedding(vocab_size, )


class MultiHeadAttention:
    def __init__(self, query, key, value, vocab_size):
        super().__init__()
        self.query_embedding = nn.Embedding(vocab_size, len(key))
        self.key_embedding = nn.Embedding(vocab_size, len(key))
        self.value_embedding = nn.Embedding(vocab_size, len(value))


    











