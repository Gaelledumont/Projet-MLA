import torch.nn as nn
from .camembert_for_pretraining import CamembertForPreTraining

class CamembertForNER(nn.Module):
    """
    Modèle CamemBERT pour la NER (classification token-level).
    """
    def __init__(self, camembert_pretrained: CamembertForPreTraining, num_labels):
        super().__init__()
        self.camembert = camembert_pretrained.camembert
        hidden_size = self.camembert.config.hidden_size
        hidden_dropout_prob = self.camembert.config.hidden_dropout_prob

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        sequence_output = self.camembert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Flatten
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss