import torch.nn as nn

class CamembertForSequenceClassification(nn.Module):
    def __init__(self, camembert_pretrained, num_labels=3):
        super().__init__()
        self.camembert = camembert_pretrained.camembert
        hidden_size = self.camembert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(self.camembert.config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.camembert(input_ids, attention_mask=attention_mask)
        # on prend le hidden state du premier token
        cls_output = outputs[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return logits, loss
