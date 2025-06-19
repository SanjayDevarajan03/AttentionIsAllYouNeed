import torch
import torch.nn as nn
from embedding import PositionalEmbedding
from transformer import Transformer
from attention import MultiHeadAttention
from NeuralNetwork import NeuralNetwork
from torchtyping import TensorType
    
class Bert(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)

    
    def forward(self,embedded: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)


    class TransformerBlack(nn.Module):

        class MultiHeadAttention(nn.Module):

            class SingleHeadAttention(nn.Module):
                def __init__(self, embedding_dim: int, attention_dim: int):
                    super().__init__()
                    torch.manual_seed(0)
                    self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
                    self.value = nn.Linear(embedding_dim, attention_dim, bias=False)
                    self.query = nn.Linear(embedding_dim, attention_dim, bias=False)

                def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                    torch.manual_seed(0)
                    key = self.key(embedded)
                    value = self.value(embedded)
                    query = self.query(embedded)

                    scores = torch.matmul(query, torch.transpose(key,1,2))
                    attention_dim = key.shape(2)

                    scores = scores/(attention_dim**0.5)
                    scores = nn.Softmax(scores, dim=2)
                    
                    return torch.round(torch.matmul(scores, value), decimals=4)
                
            
            def __init__(self, model_dim: int, num_heads: int):
                super().__init__()
                torch.manual_seed(0)
                self.att_heads = nn.ModuleList()
                for i in range(num_heads):
                    self.att_heads.append(self.SingleHeadAttention(model_dim, model_dim//num_heads))

            
            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                torch.manual_seed(0)
                head_outputs = []
                for head in self.att_heads:
                    head_outputs.append(head(embedded))

                concatenated = torch.cat(head_outputs, dim=2)
                return torch.round(concatenated, decimals=4)
            
        class NeuralNetwork(nn.Module):
            def __init__(self, model_dim: int):
                super().__init__()
                torch.manual_seed(0)
                self.up_projection = nn.Linear(model_dim, model_dim*4)
                self.relu = nn.ReLU()
                self.down_projection = nn.Linear(model_dim*4, model_dim)
                self.dropout = nn.Dropout(p=0.2)
            
            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                torch.manual_seed(0)
                return self.dropout(self.sigmoid(self.last_layer(self.first_layer(embedded))))
            
        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)

        def forward(self,):
            pass