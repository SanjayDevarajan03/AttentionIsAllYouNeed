import torch
import torch.nn as nn
from torchtyping import TensorType

class GPT(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.token_embeddings = nn.Embedding(vocab_size, model_dim)
        self.positional_embeddings = nn.Embedding(context_length, model_dim)
        self.transformerblock = nn.Sequential()
        for i in range(num_blocks):
            self.transformerblock.append(self.TransformerBlock(model_dim, num_heads))
        self.final_norm = nn.LayerNorm(model_dim)
        self.vocab_projection = nn.Linear(model_dim, vocab_size)


    def forward(self, context: TensorType[int]) -> TensorType[int]:
        torch.manual_seed(0)   
        embedded = self.token_embeddings(context)
        context_length = context.shape[1]
        positions = torch.arange(context_length)
        embedded = embedded + self.positional_embedding(positions)
        raw_output = self.vocab_projection(self.final_norm(self.transformer_blocks(embedded)))
        probabilities = nn.functional.softmax(raw_output,dim=-1)

        return torch.round(probabilities, decimals=4)


    class TransformerBlock(nn.Module):

        class MultiHeadedSelfAttention(nn.Module):
            
            class SingleHeadAttention(nn.Module):
                def __init__(self, embedding_dim, attention_dim):
                    super().__init__()
                    torch.manual_seed(0)
                    self.key = nn.Linear(embedding_dim,attention_dim, bias=False)
                    self.value = nn.Linear(embedding_dim, attention_dim, bias=False)
                    self.query = nn.Linear(embedding_dim, attention_dim, bias=False)

                def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                    torch.manual_seed(0)
                    key = self.key(embedded)
                    value = self.value(embedded)
                    query = self.query(embedded)

                    scores = torch.matmul(query, torch.transpose(key,1,2))
                    context_length, attention_dim = key.shape(1), key.shape(2)
                    scores = scores/(attention_dim**0.5)
                    lower_triangular = torch.tril(torch.ones(context_length, context_length))
                    mask = lower_triangular == 0
                    scores = scores.masked_fill(mask, float('-inf'))
                    scores = nn.functional.softmax(scores, dim=2)

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
                return concatenated
            
        class NeuralNetwork(nn.Module):
            def __init__(self, model_dim: int):
                super().__init__()
                torch.manual_seed(0)
                self.up_projection = nn.Linear(model_dim, model_dim*4)
                self.relu = nn.ReLU()
                self.down_projection = nn.Linear(model_dim*4, model_dim)
                self.dropout = nn.Dropout(0.2)

            def forward(self, x: TensorType[float]) -> TensorType[float]:
                torch.manual_seed(0)
                return self.dropout(self.down_projection(self.relu(self.up_projection(x))))
        
        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)
            self.attention = self.MultiHeadAttention(model_dim, num_heads)
            self.linear_network = self.NeuralNetwork(model_dim)
            self.first_norm = nn.LayerNorm(model_dim)
            self.second_norm = nn.LayerNorm(model_dim)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]: 
            torch.manual_seed(0)
            embedded = embedded + self.attention(self.first_norm(embedded))
            embedded = embedded + self.linear_network(self.second_norm(embedded))
            return torch.round(embedded, decimals=4)


            

