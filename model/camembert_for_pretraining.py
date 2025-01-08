import torch
import torch.nn as nn
from .camembert_model import CamembertModel, roberta_init_weights
from .camembert_config import CamembertConfig

class CamembertForPreTraining(nn.Module):
    def __init__(self, config: CamembertConfig):
        super().__init__()
        self.camembert = CamembertModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.apply(lambda module: roberta_init_weights(module, config.initializer_range))

    def forward(self, input_ids, attention_mask=None, labels=None):
        sequence_output = self.camembert(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.camembert.config.vocab_size), labels.view(-1))

        return logits, loss

    @classmethod
    def load_pretrained(cls, path, device='cuda'):
        # On charge un modèle pré-entraîné
        state = torch.load(path, map_location=device)
        config = CamembertConfig()
        model = cls(config)
        model.load_state_dict(state)
        model.to(device)
        return model