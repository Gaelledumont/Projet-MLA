import torch
import torch.nn as nn
import math

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class FFN(nn.Module):
    """
    Feed Forward Network
    Args:
    """
    def __init__(self, input_dim, hidden_dim=768, dropout=0.1, activation="gelu"):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        if activation == "gelu":
            self.act = gelu
        else:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x