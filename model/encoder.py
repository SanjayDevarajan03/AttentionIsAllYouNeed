import torch
import torch.nn as nn
from embedding import PositionalEmbedding
from transformer import Transformer
from attention import MultiHeadAttention
from NeuralNetwork import NeuralNetwork
from torchtyping import TensorType
    
class Bert(nn.Module):
    def __init__(self, model_dim: int, context_length: int, vocab_size: int, num_blocks: int, num_heads:int):
        super().__init__()
        torch.manual_seed(0)
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_embedding = nn.Embedding(context_length, model_dim)
        self.transformer_block = nn.Sequential()
        self.layer_norm = nn.LayerNorm(model_dim)
        for i in range(num_blocks):
            self.transformer_block.append(self.TransformerBlock(model_dim, num_heads))
        self.vocab_projection = nn.Linear(model_dim, vocab_size)


    
    def forward(self,context: TensorType[int]) -> TensorType[float]:
        torch.manual_seed(0)
        embedded = self.token_embedding(context)
        batch_size, context_length = context.shape
        positions = torch.arange(context_length, device=context.device).unsqueeze(0).expand(batch_size, -1)
        embedded = embedded + self.positional_embedding(positions)
        raw_output = self.vocab_projection(self.layer_norm(self.transformer_block(embedded)))
        probabilities = nn.functional.softmax(raw_output, dim=-1)
        return torch.round(probabilities, decimals=4)
    

    class TransformerBlock(nn.Module):

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
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                torch.manual_seed(0)
                return self.dropout(self.down_projection(self.relu(self.up_projection(embedded))))

            
        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)
            self.attention = self.MultiHeadAttention(model_dim, num_heads)
            self.layer_network = self.NeuralNetwork(model_dim)
            self.first_norm = nn.LayerNorm(model_dim)
            self.second_norm = nn.LayerNorm(model_dim)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            torch.manual_seed(0)
            embedded = embedded + self.attention(self.first_norm(embedded))
            embedded = embedded + self.layer_network(self.second_norm(embedded))
            return torch.round(embedded, decimals=4)

    