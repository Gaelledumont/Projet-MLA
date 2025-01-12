import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.camembert_for_pretraining import CamembertForPreTraining
from model.camembert_for_parsing import CamembertForParsing
from .parsing_dataset import ParsingDataset

@torch.no_grad()
def evaluate_parsing(model, loader, device='cuda'):
    """
    Calcul du UAS/LAS
    """
    model.eval()
    total_tokens=0
    correct_arcs=0
    correct_rels=0

    for input_ids, attention_mask, heads, rels in loader:
        input_ids=input_ids.to(device)
        attention_mask=attention_mask.to(device)
        heads=heads.to(device)
        rels=rels.to(device)
        arc_logits, rel_logits, _ = model(input_ids, attention_mask=attention_mask)
        # => pred_head = argmax sur dimension=2
        pred_heads = torch.argmax(arc_logits, dim=2)

        # gather relations
        bsz, seq_len, _ = arc_logits.size()
        rel_scores = rel_logits.permute(0,2,3,1)

        batch_idx = torch.arange(bsz).unsqueeze(-1).expand(bsz, seq_len).to(device)
        tok_idx = torch.arange(seq_len).unsqueeze(0).expand(bsz, seq_len).to(device)
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
            valid_mask = (gold_heads>=0)
            total_tokens += valid_mask.sum().item()

            arc_ok=(hyp_heads[valid_mask]==gold_heads[valid_mask])
            correct_arcs += arc_ok.sum().item()
            correct_rels += ((hyp_rels[valid_mask]==gold_rels[valid_mask]) & arc_ok).sum().item()

    uas = correct_arcs / total_tokens if total_tokens>0 else 0
    las = correct_rels / total_tokens if total_tokens>0 else 0
    model.train()
    return uas, las

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
    dropout=0.2,
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
        dropout=dropout
    ).to(device)

    # On charge les datasets
    train_data=ParsingDataset(train_path, tokenizer, rel2id, max_len=512)
    dev_data=ParsingDataset(dev_path, tokenizer, rel2id, max_len=512)
    train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader=DataLoader(dev_data, batch_size=batch_size, shuffle=False)

    # Optim
    optimizer=optim.Adam(model.parameters(),lr=lr)

    best_las=0.0
    for e in range(1, epochs+1):
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

        # On sauvegarde si c'est meilleur
        if las>best_las:
            best_las=las
            torch.save(model.state_dict(),"camembert_parsing_best.pt")
            print(f"New best model saved (LAS={las*100:.2f}%)")

    return best_las