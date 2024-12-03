import torch
import torch as nn
from Transformer import Transformer
from preprocessing.embedding import CamemBERTEmbedding

class Model(nn.Module):
    def __init__(self, device, vocab_size, max_len, input_dim, embed_dim=512, hidden_dim=768, num_heads=12, batch_size=32, dropout=0.1, layers=12):
        super(Model, self).__init__()
        self.embedding = CamemBERTEmbedding(vocab_size, max_len, embed_dim, dropout)
        self.transformer = Transformer(device, input_dim, batch_size, hidden_dim, num_heads, dropout=dropout, layers=layers)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        return x
