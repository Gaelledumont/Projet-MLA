import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from model.camembert_for_pretraining import CamembertForPreTraining
from model.camembert_for_sequence_classification import CamembertForSequenceClassification

class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, label2id, max_len=128):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                premise, hypo, lab = line.strip().split('\t')
                self.samples.append((premise, hypo, lab))
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __getitem__(self, idx):
        premise, hypo, lab = self.samples[idx]
        premise_ids = self.tokenizer.encode(premise)
        hypo_ids = self.tokenizer.encode(hypo)
        input_ids = premise_ids + [2] + hypo_ids
        # on tronque/pad
        input_ids = input_ids[:self.max_len]
        attention_mask = [1]*len(input_ids)
        while len(input_ids)<self.max_len:
            input_ids.append(1)  # pad_id
            attention_mask.append(0)
        label_id = self.label2id[lab]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(label_id, dtype=torch.long)
        )

    def __len__(self):
        return len(self.samples)

def train_nli(model_path, train_path, dev_path, tokenizer, label2id, device='cuda'):
    pretrained = CamembertForPreTraining.load_pretrained(model_path, device=device)
    model = CamembertForSequenceClassification(pretrained, num_labels=len(label2id)).to(device)

    train_dataset = NLIDataset(train_path, tokenizer, label2id)
    dev_dataset   = NLIDataset(dev_path, tokenizer, label2id)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader   = DataLoader(dev_dataset, batch_size=16, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(3):
        # train
        model.train()
        total_loss = 0
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - train loss = {total_loss/len(train_loader):.4f}")

        # dev
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
        print(f"Epoch {epoch+1} - dev acc = {acc*100:.2f}%")

    return model
