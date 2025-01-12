import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.camembert_for_pretraining import CamembertForPreTraining
from model.camembert_for_sequence_classification import CamembertForSequenceClassification
from fine_tuning.nli_trainer import NLIDataset

@torch.no_grad()
def evaluate_nli(model, test_loader, device='cuda'):
    model.eval()
    correct=0
    total=0
    for input_ids, attention_mask, labels in tqdm(test_loader):
        input_ids=input_ids.to(device)
        attention_mask=attention_mask.to(device)
        labels=labels.to(device)
        logits,_=model(input_ids, attention_mask)
        preds=torch.argmax(logits, dim=-1)
        correct += (preds==labels).sum().item()
        total += labels.size(0)
    model.train()
    return correct/total if total>0 else 0.0

def test_nli(
    best_model_path,
    base_pretrained_path,
    test_path,
    tokenizer,
    label2id,
    batch_size=16,
    device='cuda'
):
    pretrained = CamembertForPreTraining.load_pretrained(base_pretrained_path, device=device)
    num_labels = len(label2id)
    model = CamembertForSequenceClassification(pretrained, num_labels=num_labels).to(device)

    state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state)

    # On charge le dataset de test
    test_data = NLIDataset(test_path, tokenizer, label2id)
    test_loader=DataLoader(test_data, batch_size=batch_size, shuffle=False)

    acc = evaluate_nli(model, test_loader, device)
    print(f"[TEST - NLI] accuracy={acc*100:.2f}%")
    return acc