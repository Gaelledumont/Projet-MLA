import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .camembert_config import CamembertConfig

def roberta_init_weights(module: nn.Module, initializer_range: float):
    if isinstance(module, nn.Linear):
        # Poids - N(0, initializer_range)
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            # On remet le vecteur du padding_idx à zéro
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

class CamembertEmbeddings(nn.Module):
    """
    Embedding
    args :
    vocab_size (int) : taille du vocabulaire
    max_len (int) : longueur max de la séquence d'entrée (x)
    embedding_dim (int) : the size of each embedding vector
    """
    def __init__(self, config: CamembertConfig):
        super().__init__()
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=1)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        word_emb = self.word_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)

        embeddings = word_emb + pos_emb
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class CamembertSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module.
    """
    def __init__(self, config: CamembertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("Hidden size must be divisible by the number of attention heads.")
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        bsz, seq_len, hid_dim = hidden_states.size()
        q = self.query(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, hid_dim)
        return context

class CamembertSelfOutput(nn.Module):
    def __init__(self, config: CamembertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)

class CamembertAttention(nn.Module):
    def __init__(self, config: CamembertConfig):
        super().__init__()
        self.self = CamembertSelfAttention(config)
        self.output = CamembertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output

class CamembertIntermediate(nn.Module):
    def __init__(self, config: CamembertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        return hidden_states

class CamembertOutput(nn.Module):
    def __init__(self, config: CamembertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)

class CamembertLayer(nn.Module):
    def __init__(self, config: CamembertConfig):
        super().__init__()
        self.attention = CamembertAttention(config)
        self.intermediate = CamembertIntermediate(config)
        self.output = CamembertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class CamembertEncoder(nn.Module):
    """
    Full Transfomer with 12 layers
    """
    def __init__(self, config: CamembertConfig):
        super().__init__()
        self.layers = nn.ModuleList([CamembertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class CamembertModel(nn.Module):
    def __init__(self, config: CamembertConfig):
        super().__init__()
        self.config = config
        self.embeddings = CamembertEmbeddings(config)
        self.encoder = CamembertEncoder(config)

        self.apply(lambda module: roberta_init_weights(module, config.initializer_range))

    def forward(self, input_ids, attention_mask=None):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = (1 - attention_mask) * -1e4
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        embeddings = self.embeddings(input_ids)
        sequence_output = self.encoder(embeddings, attention_mask)
        return sequence_output
