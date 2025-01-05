import torch.nn as nn

class CamembertForTokenClassification(nn.Module):
    def __init__(self, camembert_for_pretraining, num_labels):
        super().__init__()
        self.camembert=camembert_for_pretraining.camembert
        hidden_size=self.camembert.config.hidden_size
        dropout_p=self.camembert.config.hidden_dropout_prob
        self.dropout=nn.Dropout(dropout_p)
        self.classifier=nn.Linear(hidden_size,num_labels)

    def forward(self,input_ids,attention_mask=None,labels=None):
        outputs=self.camembert(input_ids,attention_mask)
        # outputs shape: (bsz, seq_len, hidden_size)
        x=self.dropout(outputs)
        logits=self.classifier(x)
        loss=None
        if labels is not None:
            # On fait une CrossEntropy
            loss_fct=nn.CrossEntropyLoss(ignore_index=-100)
            loss=loss_fct(logits.view(-1,logits.size(-1)),labels.view(-1))
        return logits,loss