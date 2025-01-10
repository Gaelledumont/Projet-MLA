import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.camembert_for_pretraining import CamembertForPreTraining
from model.camembert_for_parsing import CamembertForParsing

class ParsingDataset(Dataset):
    def __init__(self, conllu_path, tokenizer, rel2id, max_len=512):
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
                splits = line.split("\t")
                if len(splits) < 4:
                    continue
                if splits[6] == '_':
                    continue
                # splits[0] => index
                # splits[1] => token
                # splits[2] => head
                # splits[3] => rel
                # on simplifie
                t = splits[1]
                h = int(splits[6])
                r = splits[7]
                tokens.append(t)
                heads.append(h)
                if r not in rel2id:
                    raise ValueError(f"Relation '{r}' inconnue")
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
        attention_mask=[]
        # On va stocker "token-level" heads & rels en un vecteur de même taille que nb tokens
        new_heads=[]
        new_rels=[]

        # On compte le nombre de tokens effectifs
        # subword alignment minimal : 1er subword => le token
        for i, (tok, hd, rl) in enumerate(zip(tokens,heads,rels)):
            subw = self.tokenizer.encode(tok)
            if len(subw) == 0:
                continue
            # le 1er subw => correspond au token i

            input_ids.extend(subw)
            # on note la HEAD
            new_heads.append(hd)
            new_rels.append(self.rel2id.get[rl])

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
        heads_out = [-1]*self.max_len
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
    n_rels,
    arc_dim=512,
    rel_dim=512,
    arc_mlp_layers=2,
    rel_mlp_layers=2,
    lr=1e-4,
    epochs=10,
    batch_size=16,
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

    train_data=ParsingDataset(train_path, tokenizer, rel2id, max_len=512)
    dev_data=ParsingDataset(dev_path, tokenizer, rel2id, max_len=512)
    train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
    dev_loader=DataLoader(dev_data,batch_size=batch_size,shuffle=False)

    optimizer=optim.Adam(model.parameters(),lr=lr)

    best_las=0.0
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
        uas,las = evaluate_parsing(model, dev_loader, rel2id, device)
        print(f"[Epoch {e}] dev UAS={uas*100:.2f}, LAS={las*100:.2f}")
        if las>best_las:
            best_las=las
            torch.save(model.state_dict(),"camembert_parsing_best.pt")
            print(f"New best model saved (LAS=(las*100:.2f)%)")

    return model

@torch.no_grad()
def evaluate_parsing(model, loader, rel2id, device='cuda'):
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
        bsz, seq_len, _ = arc_logits.size()
        rel_scores = rel_logits.permute(0,2,3,1)
        batch_idx = torch.arange(bsz).unsqueeze(-1).expand(bsz,seq_len).to(device)
        tok_idx = torch.arange(seq_len).unsqueeze(0).expand(bsz,seq_len).to(device)
        heads_clamped = pred_heads.clone()
        heads_clamped[heads_clamped<0]=0 # si jamais
        pred_rel_scores = rel_scores[batch_idx, tok_idx, heads_clamped, :]
        pred_rels = torch.argmax(pred_rel_scores, dim=2)

        # calcul
        for b in range(bsz):
            length=attention_mask[b].sum().item()
            gold_heads=heads[b,:length]
            gold_rels=rels[b,:length]
            hyp_heads=pred_heads[b,:length]
            hyp_rels=pred_rels[b,:length]
            # on ignore les positions égales à -1
            valid_mask=(gold_heads>=0)
            correct_arcs+=(hyp_heads[valid_mask]==gold_heads[valid_mask]).sum().item()
            arc_ok=(hyp_heads[valid_mask]==gold_heads[valid_mask])
            correct_rels+=((hyp_rels[valid_mask]==gold_rels[valid_mask]) & arc_ok).sum().item()
            total_tokens+=valid_mask.sum().item()

    uas = correct_arcs / total_tokens if total_tokens>0 else 0
    las = correct_rels / total_tokens if total_tokens>0 else 0
    model.train()
    return uas, las