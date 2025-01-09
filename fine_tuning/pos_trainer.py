import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.camembert_for_pretraining import CamembertForPreTraining
from model.camembert_for_token_classification import CamembertForTokenClassification

class POSDataset(Dataset):
    def __init__(self, conll_path, tokenizer, label2id, max_len=128):
        # On charge tout en mémoire ou en stream
        self.samples=[]
        with open(conll_path,"r",encoding="utf-8") as f:
            tokens=[]
            labels=[]
            for line in f:
                line=line.strip()
                if not line:
                    if tokens:
                        self.samples.append((tokens,labels))
                        tokens,labels=[],[]
                    continue
                splits = line.split("\t")
                if len(splits) < 4 or line.startswith("#"):  # Ignore les lignes de commentaire ou malformées
                    continue
                t = splits[1]  # Token
                lab = splits[3]  # UPOS
                tokens.append(t)
                labels.append(lab)
            if tokens:
                self.samples.append((tokens,labels))
        self.tokenizer=tokenizer
        self.label2id=label2id
        self.max_len=max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        tokens, labels=self.samples[idx]
        input_ids=[2]
        label_ids=[-100]
        for t,lab in zip(tokens,labels):
            subids=self.tokenizer.encode(t)
            if not subids:
                continue
            lab_id=self.label2id.get(lab,self.label2id["O"])
            input_ids+=subids
            # le premier sub-token reçoit le label, le reste -100
            label_ids+=[lab_id]+([-100]*(len(subids)-1))
        # On tronque
        input_ids=input_ids[:self.max_len]
        label_ids=label_ids[:self.max_len]
        # On pad
        while len(input_ids)<self.max_len:
            input_ids.append(self.tokenizer.pad_token_id())
            label_ids.append(-100)
        attn_mask=[1 if x!=self.tokenizer.pad_token_id() else 0 for x in input_ids]
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
    lr=3e-5,
    epochs=3,
    batch_size=16,
    device="cuda"
):
    # On charge le modèle
    pretrained=CamembertForPreTraining.load_pretrained(pretrained_path,device=device)
    model=CamembertForTokenClassification(pretrained, num_labels).to(device)

    train_data=POSDataset(train_path,tokenizer,label2id)
    dev_data=POSDataset(dev_path,tokenizer,label2id)
    train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
    dev_loader=DataLoader(dev_data,batch_size=batch_size,shuffle=False)

    optimizer=optim.Adam(model.parameters(),lr=lr)

    for e in range(1,epochs+1):
        model.train()
        total_loss=0.0
        for input_ids,attn_mask,labels in tqdm(train_loader,desc=f"Epoch {e}"):
            input_ids=input_ids.to(device)
            attn_mask=attn_mask.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            _,loss=model(input_ids,attn_mask,labels=labels)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        avg_loss=total_loss/len(train_loader)
        print(f"Epoch {e} - train loss={avg_loss:.4f}")

        # On évalue
        model.eval()
        correct, total=0,0
        with torch.no_grad():
            for input_ids,attn_mask,labels in dev_loader:
                input_ids=input_ids.to(device)
                attn_mask=attn_mask.to(device)
                labels=labels.to(device)
                logits,_=model(input_ids,attn_mask)
                preds=torch.argmax(logits,dim=-1)
                mask=(labels!=-100)
                correct+=(preds[mask]==labels[mask]).sum().item()
                total+=mask.sum().item()
        acc=correct/total if total>0 else 0
        print(f"Epoch {e} - dev acc={acc*100:.2f}%")

    # On sauvegarde
    torch.save(model.state_dict(),"camembert_pos_finetuned.pt")
    return model