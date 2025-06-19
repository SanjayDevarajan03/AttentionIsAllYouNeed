import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
import numpy as np





class MultiHeadAttention:
    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.att_heads = nn.ModuleList()
        for i in range(num_heads):
            self.att_heads.append(self.SingleHeadAttention(embedding_dim, attention_dim//num_heads))
        
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        head_outputs = []
        for head in self.att_heads:
            head_outputs.append(head(embedded))
        concatenated = torch.cat(head_outputs, dim=2)
        return torch.round(concatenated, decimals=4)

    class SingleHeadAttention:
        def __init__(self, embedding_dim, attention_dim):
            super().__init__()
            torch.manual_seed(0)

            self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.value = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.query = nn.Linear(embedding_dim, attention_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            key = self.key(embedded)
            query = self.query(embedded)
            value = self.value(embedded)

            scores = torch.matmul(query, torch.transpose(key,1,2))
            context_length, attention_dim = key.shape[1], key.shape[2]
            scores = scores/(attention_dim**0.5)

            lower_triangular = torch.tril(torch.ones(context_length, context_length))
            mask = lower_triangular == 0
            scores = scores.masked_fill(mask, float('-inf'))
            scores = nn.Softmax(dim=2)(scores)

            return torch.round(torch.matmul(scores, value), decimals=4)


