import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from .camembert_for_pretraining import CamembertForPreTraining
from model.camembert_for_sequence_classification import CamembertForSequenceClassification

class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_path, tokenizer, label2id, max_len=512):
        self.samples = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            # Skip header line
            next(f) # on saute l'en-tête
            for line in f:
                line = line.strip()
                if not line:
                    continue
                splits = line.split("\t")
                if len(splits) != 3:
                    # on ignore
                    continue
                premise, hypo, lab = splits
                if lab not in label2id:
                    raise ValueError(f"Label NLI inconnu : {lab}")
                self.samples.append((premise, hypo, lab))
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        premise, hypo, lab = self.samples[idx]
        premise_ids = self.tokenizer.encode(premise)
        hypo_ids = self.tokenizer.encode(hypo)
        input_ids = [2] + premise_ids + [2, 2] + hypo_ids + [2]
        # on tronque/pad
        if len(input_ids)>self.max_len:
            input_ids = input_ids[:self.max_len]
        attention_mask = [1]*len(input_ids)
        while len(input_ids)<self.max_len:
            input_ids.append(self.tokenizer.pad_token_id)  # pad_id
            attention_mask.append(0)

        label_id = self.label2id[lab]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(label_id, dtype=torch.long)
        )

def train_nli(
        model_path,
        train_path,
        dev_path,
        tokenizer,
        label2id,
        num_labels,
        epochs=3,
        lr=1e-5,
        batch_size=32,
        device='cuda',
        out_model_path=None,
):
    """
    Fine-tuning NLI.
    """
    # 1) On charge le modèle pré-entraîné
    pretrained = CamembertForPreTraining.load_pretrained(model_path, device=device)
    # 2) On instancie CamemBERT pour la classification de séquences
    model = CamembertForSequenceClassification(pretrained, num_labels=num_labels).to(device)

    # 3) On charge les datasets
    train_dataset = NLIDataset(train_path, tokenizer, label2id, max_len=512)
    dev_dataset = NLIDataset(dev_path, tokenizer, label2id, max_len=512)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc=0.0
    best_model_state=None

    for epoch in range(1, epochs+1):
        # train
        model.train()
        total_loss = 0.0
        for input_ids, attention_mask, labels in tqdm(train_loader,desc=f"Epoch {epoch}"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss/len(train_loader)
        print(f"[Epoch {epoch}] train loss = {avg_loss:.4f}")

        # eval sur dev
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in dev_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                logits, _ = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds==labels).sum().item()
                total += labels.size(0)
        acc = correct/total if total>0 else 0
        print(f"[Epoch {epoch}] dev acc = {acc*100:.2f}%")

        if acc>best_acc:
            best_acc=acc
            best_model_state=model.state_dict()
            print(f"New best dev acc={acc*100:.2f}%")

    # on recharge la best
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if out_model_path and best_model_state is not None:
        torch.save(model.state_dict(), out_model_path)
        print(f"Best model saved => {out_model_path} (acc={best_acc*100:.2f}%)")

    return best_acc