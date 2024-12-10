import torch
import torch.nn as nn


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
    def __init__(self, device, dim, batch_size, num_heads=12, qkv_bias=False):
        super().__init__()
        self.device = device
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = int(dim / num_heads)

        self.scale = self.head_dim ** -0.5    
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias) #Queries, Keys, Values
        self.batch_size = batch_size
        self.proj = nn.Linear(dim, dim) 

    def forward(self, x):

        x = x.to(self.device)

        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x)  # Shape: (batch_size, N, 3 * dim)
        
        qkv = qkv.to(self.device).view(3, batch_size, self.num_heads, num_tokens, self.head_dim)  # Shape: (batch_size, N, 3, num_heads, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shapes: (batch_size, num_heads, N, head_dim)
        
        attn = (q * self.scale) @ k.transpose(-2, -1) 

        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1,2).reshape(batch_size, num_tokens, channels).to(self.device)

        x = self.proj(x)

        return x
