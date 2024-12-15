import torch
import torch.nn as nn
from Transformer import TransformerEncoder
from embedding import CamemBERTEmbedding

class CamemBERTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = CamemBERTEmbedding(config)
        self.encoder = TransformerEncoder(config)

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.embedding(input_ids)
        hidden_states = self.encoder(hidden_states, attention_mask=extended_attention_mask)
        return hidden_states