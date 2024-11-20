import torch
import torch as nn
from MHSA import MultiHeadSelfAttention
from FFN import FFN

class TransformerLayer(nn.Module):
    """
    One layer of the transformer
    args : 

    """
    def __init__(self, device, input_dim, batch_size, hidden_dim=768, num_heads=12, qkv_bias=False, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.FFN = FFN(input_dim, hidden_dim, dropout)
        self.MHSA = MultiHeadSelfAttention(device, input_dim, batch_size, num_heads, qkv_bias)
        self.layerNorm1 = nn.LayerNorm(input_dim)
        self.layerNorm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn = self.MHSA(x)
        x = x + self.dropout(attn)
        x = self.layerNorm1(x)

        ffn = self.FFN(x)
        x = x + self.dropout(ffn)
        x = self.layerNorm2(x)

        return x

class Transformer(nn.Module):
    """
    Full Transfomer with 12 layers
    """
    def __init__(self, device, input_dim, batch_size, hidden_dim=768, num_heads=12, qkv_bias=False, dropout=0.1, layers=12):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(device, input_dim, batch_size, hidden_dim, num_heads, qkv_bias, dropout)
            for _ in range(layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x