import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.camembert_for_pretraining import CamembertForPreTraining
from model.camembert_for_token_classification import CamembertForTokenClassification
from fine_tuning.pos_trainer import POSDataset # on reutilise la classe du training

def test_pos(
    best_model_path,
    base_pretrained_path,
    test_path,
    tokenizer,
    label2id,
    i2dlabel,
    batch_size=16,
    device="cuda",
):
    """
    Charge le meilleur modèle POS, charge le test set, calcule l'accuracy (oken-level).
    """
    # 1) On charge le modèle pré-entraîné + la tête fine-tunée

    # On charge le modèle pré-entraîné
    pretrained = CamembertForPreTraining.load_pretrained(base_pretrained_path, device=device)
    num_labels = len(label2id)
    model = CamembertForTokenClassification(pretrained, num_labels=num_labels).to(device)

    # On charge le modèle fine-tuné
    state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state)

    # 2) On charge le test dataset
    test_data = POSDataset(test_path, tokenizer, label2id) # même classe que pour le train
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # 3) On évalue
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs_ids, attention_mask, labels in tqdm(test_loader):
            inputs_ids = inputs_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits, _ = model(inputs_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
            mask = (labels != -100)
            correct += (preds[mask] == labels[mask]).sum().item()
            total += mask.sum().item()

    acc = correct / total if total > 0 else 0
    print(f"[TEST - POS] accuracy = {acc * 100:.2f}%")
    return acc