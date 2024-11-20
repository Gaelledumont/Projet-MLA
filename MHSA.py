import torch
import torch.nn as nn

#the number of attention heads must evenly divide the number of channels
class MultiHeadSelfAttention(nn.Module):   
    """
    Multi-Head Self-Attention module.

    Computes self-attention for multiple heads with relative positional encoding.

    Args:
        num_heads (int): Number of attention heads.
        dim (int): Dimensionality of the input features.
        batch_size (int): Batch size of the input tensor.
        window_size (tuple of int, optional): Size of the attention window. Default is (8, 8).
        qkv_bias (bool, optional): Whether to include a bias term in the Q, K, and V linear layers. Default is False.

    Attributes:
        num_heads (int): Number of attention heads.
        dim (int): Dimensionality of the input features.
        head_dim (int): Dimensionality of each attention head.
        scale (float): Scaling factor for the attention scores.
        qkv (nn.Linear): Linear layer for projecting input features to queries, keys, and values.
        window_size (tuple of int): Size of the attention window.
        relative_position_bias_table (nn.Parameter): Table of relative position biases.
        relative_position_index (torch.Tensor): Index tensor for relative position encoding.


    Methods:
        forward(x): Computes the self-attention for the input and returns the attended feature map.
    """     
    def __init__(self, device, num_heads, dim, batch_size, window_size=(8,8), qkv_bias=False):
        super().__init__()
        self.device = device
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = int(dim / num_heads)
        self.scale = self.head_dim ** -0.5    
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias) #Queries, Keys, Values
        self.window_size = window_size
        self.batch_size = batch_size

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  
        coords_flatten = torch.flatten(coords, 1)  
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  
        relative_coords[:, :, 0] += self.window_size[0] - 1  
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):

        x = x.to(self.device)
        relative_position_index = self.relative_position_index.to(self.device)

        num_windows_B, num_tokens, channels = x.shape
        qkv = self.qkv(x)  # Shape: (num_windows * B, N, 3 * dim)
        
        qkv = qkv.to(self.device).view(3, num_windows_B, self.num_heads, num_tokens, self.head_dim)  # Shape: (num_windows * B, N, 3, num_heads, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shapes: (num_windows * B, num_heads, N, head_dim)
        
        attn = (q * self.scale) @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(-1, self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1])  

        relative_position_bias = relative_position_bias.contiguous().to(self.device)
        
        attn = attn.view(num_tokens*self.batch_size//16, self.num_heads, self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1])  #num_tokens * batch_size /16 ??
        
        attn = attn.to(self.device) + relative_position_bias.unsqueeze(0)   

        attn = attn.softmax(dim=-1)
        
        attn = attn.view(num_windows_B, self.num_heads, num_tokens, num_tokens).to(self.device)
        x = (attn @ v).transpose(1,2).reshape(num_windows_B, num_tokens, channels).to(self.device)

        return x