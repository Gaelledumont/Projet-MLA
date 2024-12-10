import torch
import torch.nn as nn

class FFN(nn.Module):
    """
    Feed Forward Network
    Args:
    """
    def __init__(self, input_dim, hidden_dim=768, dropout=0.1):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x



