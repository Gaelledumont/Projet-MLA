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
    def __init__(self, config):
        super(CamemBERTEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id) # vecteur qui possède le 'sens' du mot
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size) # vecteur qui possède la position du mot
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids):
        seq_length = input_ids.size(1)  # Longueur de la séquence
        position_ids = self.position_ids[:, :seq_length]

        word_embeddings = self.token_embedding(input_ids)
        pos_embeddings = self.position_embedding(position_ids)

        embeddings = word_embeddings + pos_embeddings
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