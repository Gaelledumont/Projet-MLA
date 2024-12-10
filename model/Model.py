import torch
import torch.nn as nn
from Transformer import Transformer
from embedding import CamemBERTEmbedding

class Model(nn.Module):
    def __init__(self, device, vocab_size, max_len, input_dim, embed_dim=768, hidden_dim=768, num_heads=12, batch_size=32, dropout=0.1, layers=12):
        super(Model, self).__init__()
        self.embedding = CamemBERTEmbedding(vocab_size, max_len, embed_dim, dropout)
        self.transformer = Transformer(device, input_dim, batch_size, hidden_dim, num_heads, dropout=dropout, layers=layers)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        return x

batch_size = 32
max_len = 128
vocab_size = 10000  # Exemple de vocabulaire
embed_dim = 768
input_dim = 768  # La dimension de l'input pour chaque token
hidden_dim = 768  # Dimension de l'espace caché
num_heads = 12  # Nombre de têtes pour l'attention
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Générer des 'dummy' values pour l'input_ids (tokens aléatoires)
input_ids = torch.randint(0, vocab_size, (batch_size, max_len)).to(device)
print('input_ids', input_ids.shape)
model=Model(device, vocab_size, max_len, input_dim)
out = model(input_ids)

# Tester le modèle
print("OUTPUT", out.shape) 