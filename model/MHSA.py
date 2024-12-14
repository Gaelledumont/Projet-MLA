import torch
import torch.nn as nn
import math


"""l'attention c'est ce qui dit au modèle 'concentre toi la dessu s'cest important' c'est calculé grâce à 3 matries : Queries, Keys et Values
+ de détails slides 70-72 dans el cours d'Obin
ou https://medium.com/@geetkal67/attention-networks-a-simple-way-to-understand-self-attention-f5fb363c736d
"""

#the number of attention heads must evenly divide the number of channels
class MultiHeadSelfAttention(nn.Module):   
    """
    Multi-Head Self-Attention module.

    Args:
        num_heads (int): Number of attention heads.
        dim (int): Dimensionality of the input features.
        batch_size (int): Batch size of the input tensor.
        qkv_bias (bool, optional): Whether to include a bias term in the Q, K, and V linear layers. Default is False.
    """     
    def __init__(self, device, dim, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5    
        self.qkv = nn.Linear(dim, dim*3, bias=True) #Queries, Keys, Values
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        # x: (batch, seq_len, dim)
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Shapes: (batch_size, num_heads, N, head_dim)

        q = q.transpose(1, 2) # [batch, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q * self.scale) @ k.transpose(-1, -2) # [batch, num_heads, seq_len, seq_len]
        if attention_mask is not None:
            attn += attention_mask

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v # [batch, num_heads, seq_len, head_dim)
        out = out.transpose(1,2).contigous().reshape(batch_size, seq_len, self.dim)
        out = self.proj(out)

        return out