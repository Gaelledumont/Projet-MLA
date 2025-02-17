import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from Camembert_4GB_model.camembert_for_pretraining import CamembertForPreTraining
from model.camembert_for_token_classification import CamembertForTokenClassification

class POSDataset(Dataset):
    def __init__(self, conll_path, tokenizer, label2id, max_len=512):
        # On charge tout en mémoire ou en stream
        self.samples=[]
        with open(conll_path, "r", encoding="utf-8") as f:
            tokens=[]
            labels=[]
            for line in f:
                line=line.strip()
                if not line:
                    if tokens:
                        self.samples.append((tokens, labels))
                        tokens, labels=[],[]
                    continue
                splits = line.split("\t")
                if len(splits)>=2:
                    t = splits[1]  # Token
                    lab = splits[3]  # UPOS
                    tokens.append(t)
                    if lab == '_':
                        # on ignore les tokens sans UPOS
                        continue
                    # si lab n'existe pas dans label2id, on lève une exception
                    if lab not in label2id:
                        raise ValueError(f"Unknow POS label '{lab}'")
                    labels.append(lab)
            if tokens:
                self.samples.append((tokens, labels))

        self.tokenizer=tokenizer
        self.label2id=label2id
        self.max_len=max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, labels = self.samples[idx]
        input_ids = [2] # <s> roberta
        label_ids=[-100]
        for t, lab in zip(tokens, labels):
            subids=self.tokenizer.encode(t)
            if not subids:
                continue
            lab_id=self.label2id[lab] # on a vérifié plus haut qu'il existe
            input_ids += subids
            # le premier sub-token reçoit le label, le reste -100
            label_ids += [lab_id] + ([-100]*(len(subids)-1))
        # On tronque
        input_ids=input_ids[:self.max_len]
        label_ids=label_ids[:self.max_len]
        # On pad
        while len(input_ids)<self.max_len:
            input_ids.append(self.tokenizer.pad_token_id)
            label_ids.append(-100)
        attn_mask=[1 if x!=self.tokenizer.pad_token_id else 0 for x in input_ids]
        return (
            torch.tensor(input_ids,dtype=torch.long),
            torch.tensor(attn_mask,dtype=torch.long),
            torch.tensor(label_ids,dtype=torch.long)
        )

def train_pos(
    pretrained_path,
    train_path,
    dev_path,
    tokenizer,
    label2id,
    num_labels,
    lr,
    max_epochs=10,
    batch_size=16,
    device="cuda",
    out_model_path=None
):
    """
    Fine-tune CamembertForTokenClassification sur le POS tagging jusqu'à 30 époques.
    On renvoie la meilleure accuracy dev rencontrée
    """
    # 1) On charge le modèle
    pretrained=CamembertForPreTraining.load_pretrained(pretrained_path, device=device)
    # 2) On crée la tête POS
    model=CamembertForTokenClassification(pretrained, num_labels).to(device)

    # 3) On charge les datasets
    train_data=POSDataset(train_path, tokenizer, label2id, max_len=512)
    dev_data=POSDataset(dev_path, tokenizer, label2id, max_len=512)
    train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
    dev_loader=DataLoader(dev_data,batch_size=batch_size,shuffle=False)

    # 4) Optim
    optimizer=optim.Adam(model.parameters(),lr=lr)

    best_acc=0.0
    best_model_state=None

    for e in range(1, max_epochs+1):
        # train
        model.train()
        total_loss=0.0
        for input_ids, attn_mask, labels in tqdm(train_loader, desc=f"Epoch {e}"):
            input_ids=input_ids.to(device)
            attn_mask=attn_mask.to(device)
            labels=labels.to(device)

            optimizer.zero_grad()
            logits, loss=model(input_ids,attn_mask,labels=labels)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()

        avg_loss=total_loss/len(train_loader)
        print(f"[Epoch {e}] train loss={avg_loss:.4f}")

        # On évalue
        model.eval()
        correct, total=0,0
        with torch.no_grad():
            for input_ids, attn_mask, labels in dev_loader:
                input_ids=input_ids.to(device)
                attn_mask=attn_mask.to(device)
                labels=labels.to(device)
                logits,_=model(input_ids, attn_mask)
                preds=torch.argmax(logits,dim=-1)
                mask=(labels!=-100)
                correct+=(preds[mask]==labels[mask]).sum().item()
                total+=mask.sum().item()
        acc=correct/total if total>0 else 0
        print(f"[Epoch {e}] dev acc={acc*100:.2f}%")

        # On sauvegarde
        if acc>best_acc:
            best_acc=acc
            best_model_state=model.state_dict()
            print(f"New best dev acc={acc*100:.2f}% (epoch={e})")

    # restore best at end
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if out_model_path and best_model_state is not None:
        torch.save(model.state_dict(), out_model_path)
        print(f"Best model saved to {out_model_path} (acc={best_acc*100:.2f}%)")

    return best_acc