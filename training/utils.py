import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Fixe la graine aléatoire pour Python, NumPy et PyTorch (CPU et CUDA)
    afin d'améliorer la reproductibilité.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # options PyTorch pour le déterminisme
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_dev_perplexity(model, dev_dataloader, device='cuda'):
    """
    Calcule la perplexité sur un dev_dataloader (MLM).
    On suppose que dev_dataloader renvoie (input_ids, attention_mask, labels).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in dev_dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # forward
            _, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            # On calcule le nb de tokens "effectifs" pour la normalisation
            # en comptant les positions != -100 (tokens masqués)
            mask = (labels != -100).long()
            n_tokens = mask.sum().item()

            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    model.train()  # repasse le modèle en mode train
    if total_tokens == 0:
        return float('inf')

    avg_loss = total_loss / total_tokens
    ppl = np.exp(avg_loss)
    return ppl