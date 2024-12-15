import torch
import torch.nn as nn
from CamemBERTModel import CamemBERTModel

class CamemBERTForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.camembert = CamemBERTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.lm_head.bias = self.lm_bias

    def forward(self, input_ids, labels=None):
        hidden_states = self.camembert(input_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return logits, loss