import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        loss = self.CELoss(preds, targets)

        return loss