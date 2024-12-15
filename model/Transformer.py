import torch.nn as nn
from MHSA import MultiHeadSelfAttention
from FFN import FFN

class TransformerLayer(nn.Module):
    """
    Two layers of the transformer
    args :

    """
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config.hidden_size, config.num_attention_heads, config.dropout_prob)
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.ffn = FFN(config.hidden_size, config.intermediate_size, config.dropout_prob, config.hidden_act)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x, attention_mask=None):
        attn_out = self.attention(x, attention_mask)
        x = x + self.dropout(attn_out)
        x = self.layerNorm1(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.layernorm2(x)
        return x

class TransformerEncoder(nn.Module):
    """
    Full Transfomer with 12 layers
    """
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return hidden_states
    
"""
In order to ob-
tain a representation for a given token, we first
compute the average of each sub-word's represen-
tations in the last four layers of the Transformer,
and then average the resulting sub-word vectors."""