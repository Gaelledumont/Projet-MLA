import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.camembert_for_pretraining import CamembertForPreTraining
from model.camembert_for_parsing import CamembertForParsing

class ParsingDataset(Dataset):
    def __init__(self, conllu_path, tokenizer, rel2id, max_len=128):
        # On parse et on stocke
        self.samples = []
        with open(conllu_path, "r", encoding="utf-8") as f:
            tokens = []
            heads = []
            rels = []
            for line in f:
                line=line.strip()
                if not line:
                    if tokens:
                        self.samples.append((tokens, heads, rels))
                        tokens, heads, rels=[],[],[]
                    continue
                splits=line.split("\t")
                # splits[0] => index
                # splits[1] => token
                # splits[2] => head
                # splits[3] => rel
                # on simplifie
                t = splits[1]
                h = int(splits[2])
                r = splits[3]
                tokens.append(t)
                heads.append(h)
                rels.append(r)
            if tokens:
                self.samples.append((tokens, heads, rels))
        self.tokenizer=tokenizer
        self.rel2id=rel2id
        self.max_len=max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, heads, rels = self.samples[idx]
        # on encode les tokens en subwords
        # On va faire la concaténation subwords de chaque token => input_ids
        input_ids=[2]
        new_heads=[]
        new_rels=[]
        for i, (tok, hd, rl) in enumerate(zip(tokens,heads,rels)):
            subw = self.tokenizer.encode(tok)
            if not subw:
                continue
            # le 1er subw => correspond au token i

            input_ids.extend(subw)
            # on note la HEAD
            new_heads.append(hd)
            new_rels.append(self.rel2id.get(rl,0))

        # On tronque
        input_ids = input_ids[:self.max_len]
        attn_mask = [1]*len(input_ids)
        while len(input_ids)<self.max_len:
            input_ids.append(self.tokenizer.pad_token_id())
            attn_mask.append(0)

        # On convertit new_heads en un array de longueur = nb tokens
        # => On a potentiellement plus de subwords que de tokens
        # => On va faire heads = array de length = len(new_heads),

        # on pad
        # si c'est plus grand que max_len, c'est un corner case
        tok_count = len(new_heads)
        heads_out = [0]*self.max_len
        rels_out  = [0]*self.max_len
        for i in range(min(tok_count, self.max_len)):
            heads_out[i] = new_heads[i]
            rels_out[i]  = new_rels[i]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attn_mask, dtype=torch.long),
            torch.tensor(heads_out, dtype=torch.long),
            torch.tensor(rels_out,  dtype=torch.long)
        )

def train_parsing(
    pretrained_path,
    train_path,
    dev_path,
    tokenizer,
    rel2id,
    arc_dim=512,
    rel_dim=512,
    arc_mlp_layers=2,
    rel_mlp_layers=2,
    n_rels=30,
    lr=1e-4,
    epochs=30,
    batch_size=32,
    device='cuda'
):
    # On charge Camembert pré-entraîné
    pretrained = CamembertForPreTraining.load_pretrained(pretrained_path, device=device)
    model = CamembertForParsing(
        camembert_pretrained=pretrained,
        arc_dim=arc_dim,
        rel_dim=rel_dim,
        n_rels=n_rels,
        arc_mlp_layers=arc_mlp_layers,
        rel_mlp_layers=rel_mlp_layers,
        dropout=0.2
    ).to(device)

    train_data=ParsingDataset(train_path, tokenizer, rel2id)
    dev_data=ParsingDataset(dev_path, tokenizer, rel2id)
    train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
    dev_loader=DataLoader(dev_data,batch_size=batch_size,shuffle=False)

    optimizer=optim.Adam(model.parameters(),lr=lr)

    for e in range(1,epochs+1):
        model.train()
        total_loss=0.0
        for input_ids, attention_mask, heads, rels in tqdm(train_loader,desc=f"Epoch {e}"):
            input_ids=input_ids.to(device)
            attention_mask=attention_mask.to(device)
            heads=heads.to(device)
            rels=rels.to(device)

            optimizer.zero_grad()
            arc_logits, rel_logits, loss=model(input_ids, attention_mask, heads, rels)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        avg_loss=total_loss/len(train_loader)
        print(f"[Epoch {e}] train loss={avg_loss:.4f}")

        # On évalue
        uas,las = evaluate_parsing(model, dev_loader, device)
        print(f"[Epoch {e}] dev UAS={uas*100:.2f}, LAS={las*100:.2f}")

    torch.save(model.state_dict(),"camembert_parsing_finetuned.pt")
    return model

@torch.no_grad()
def evaluate_parsing(model, loader, device):
    model.eval()
    total_tokens=0
    correct_arcs=0
    correct_rels=0
    for input_ids, attention_mask, heads, rels in loader:
        input_ids=input_ids.to(device)
        attention_mask=attention_mask.to(device)
        heads=heads.to(device)
        rels=rels.to(device)
        arc_logits, rel_logits, _=model(input_ids, attention_mask=attention_mask)
        # => pred_head = argmax sur dimension=2
        pred_heads = torch.argmax(arc_logits, dim=2)
        # gather
        bsz, seq_len, seq_len_ = arc_logits.size()
        heads_ = pred_heads
        rel_scores = rel_logits.permute(0,2,3,1)
        batch_idx = torch.arange(bsz).unsqueeze(-1).expand(bsz,seq_len).to(device)
        tok_idx   = torch.arange(seq_len).unsqueeze(0).expand(bsz,seq_len).to(device)
        pred_rel_scores = rel_scores[batch_idx, tok_idx, heads_, :]
        pred_rels = torch.argmax(pred_rel_scores, dim=2)

        # mask out pad
        mask = (attention_mask==1)
        correct_arcs += ((pred_heads==heads) & mask).sum().item()
        correct_rels += ((pred_heads==heads) & (pred_rels==rels) & mask).sum().item()
        total_tokens += mask.sum().item()

    uas = correct_arcs / total_tokens if total_tokens>0 else 0
    las = correct_rels / total_tokens if total_tokens>0 else 0
    return uas, las