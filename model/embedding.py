import torch
import torch.nn as nn

# https://www.youtube.com/watch?v=dichIcUZfOw (pourquoi on fait du positional encoding, 
#le reste de la vidéo est moins important parce que c'est pas le même encding qu'est utilisé)

class CamemBERTEmbedding(nn.Module):
    """
    Embedding
    args :
    vocab_size (int) : taille du vocabulaire : tous les mots qu'on connait (10 000 actuellement si je dis pas de bêtises)
    max_len (int) : longueur max de la séquence d'entrée (x)
    embedding_dim (int) : the size of each embedding vector
    """
    def __init__(self, vocab_size, max_len, embed_dim=768, dropout=0.1):
        super(CamemBERTEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim) # vecteur qui possède le 'sens' du mot
        self.position_embedding = nn.Embedding(max_len, embed_dim) # vecteur qui possède la position du mot
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout) 

    def forward(self, input_ids):
        seq_len = input_ids.size(1)  # Longueur de la séquence
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)  # (1, seq_len) pour être compatible avec input_ids qui a le batch_size

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)

        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings



"""
d'abord on a créé un tokenizer qui prend un corpus de mots et crée une sorte de dictionnaire où il associe chaque mot à une valeur, un id, 
ensuite on va prendre notre phrase à encoder, la transofmer en token (donc valeur des mots correspondants dasn le 'dictionnaire') 
on va créer des vecteurs d'embeddings à nos tokens qui représentent le 'sens' du mot, initialisés d'abord aléatoirement grâce à nn.Embedding, 
qui passe par un apprentissage (donc notre modèle transformer) et grâce à la backpropagation, on va recalculer nos vecteurs d'embeddding de sorte 
que des mtos tel que 'chat' et 'félin' auront des valeurs de vecteurs assez proches. 
parallèlement on décide de créer des vecteurs d'embedding pour nos tokens qui ne donnent pas une info sur le 'sens' du mot mais cette fois ci 
sur sa position dans la phrase car c'est important de garder le contexte pour faire de meilleures prédictions"""