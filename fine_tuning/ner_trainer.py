import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.camembert_for_pretraining import CamembertForPreTraining
from model.camembert_for_ner import CamembertForNER

# 1) On extrait des entités sur un schéma BIO
def extract_entities_bio(labels):
    """
    Reçoit une liste de labels, ex: ["O","B-PER","I-PER","O","B-LOC","I-LOC"].
    Retourne une liste d'entités (start, end, type), où 'end' est inclusif.
    """
    entities = []
    start = None
    ent_type = None

    for i, lab in enumerate(labels):
        if lab.startswith("B-"):
            # on ferme l'éventuelle entité précédente
            if start is not None:
                entities.append((start, i-1, ent_type))
            start = i
            ent_type = lab[2:]
        elif lab.startswith("I-"):
            # si on n'était pas en entité, on peut soit forcer un B-, soit ignorer
            # on simplifie en disant : si start is None, on commence ici
            if start is None:
                start = i
                ent_type = lab[2:]
            # sinon, on continue la même entité
        else:
            # "O" ou autre
            if start is not None:
                entities.append((start, i-1, ent_type))
                start = None
                ent_type = None

    # fin de séquence
    if start is not None:
        entities.append((start, len(labels)-1, ent_type))
    return entities

def compute_f1_bio(pred_labels, gold_labels):
    """
    pred_labels, gold_labels : listes de labels (BIO) sur toute une phrase.
    On extrait les entités (start, end, type) pour chaque, on calcule l'intersection
    => precision, recall, f1.
    """
    pred_ents = set(extract_entities_bio(pred_labels))
    gold_ents = set(extract_entities_bio(gold_labels))
    tp = len(pred_ents & gold_ents)
    fp = len(pred_ents - gold_ents)
    fn = len(gold_ents - pred_ents)

    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return tp, fp, fn, prec, rec, f1

# 2) Dataset pour la NER
class NERDataset(Dataset):
    def __init__(self, conll_path, tokenizer, label2id, max_len=128):
        self.samples = []
        with open(conll_path, 'r', encoding='utf-8') as f:
            tokens, labels = [], []
            for line in f:
                line = line.strip()
                if not line:
                    if tokens:
                        self.samples.append((tokens, labels))
                        tokens, labels = [], []
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    tok = parts[0]
                    lab = parts[1]
                    tokens.append(tok)
                    labels.append(lab)
            # fin du fichier
            if tokens:
                self.samples.append((tokens, labels))

        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, labels = self.samples[idx]
        # Encodage subwords
        # on insère un <s> (id=2) au début
        input_ids = [2]
        label_ids = [-100]

        for t, lab in zip(tokens, labels):
            sub_ids = self.tokenizer.encode(t)
            if not sub_ids:
                continue
            # label
            lab_id = self.label2id.get(lab, self.label2id["O"]) # plan B
            # le premier sub-token reçoit le label, le reste -100
            input_ids += sub_ids
            label_ids += [lab_id] + [-100]*(len(sub_ids)-1)

        # On tronque
        input_ids = input_ids[:self.max_len]
        label_ids = label_ids[:self.max_len]

        # On pad
        while len(input_ids) < self.max_len:
            input_ids.append(self.tokenizer.pad_token_id())
            label_ids.append(-100)

        attention_mask = [1 if x!=self.tokenizer.pad_token_id() else 0 for x in input_ids]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(label_ids, dtype=torch.long)
        )

# 3) On évalue sur un set NER (F1 BIO)
def evaluate_ner(model, dev_loader, id2label, device='cuda'):
    """
    Calcul d'un F1 exact (BIO) sur le dev_loader.
    """
    model.eval()
    tp_all, fp_all, fn_all = 0, 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in dev_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits, _ = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1)

            bsz = input_ids.size(0)
            for b in range(bsz):
                # longueur effective
                length = attention_mask[b].sum().item()
                gold_seq = labels[b, :length].cpu().tolist()
                pred_seq = preds[b, :length].cpu().tolist()

                # on filtre les positions
                filtered_gold = []
                filtered_pred = []
                for g, p in zip(gold_seq, pred_seq):
                    if g != -100:
                        filtered_gold.append(g)
                        filtered_pred.append(p)

                # on convertit en string
                gold_labels_str = [id2label[gx] for gx in filtered_gold]
                pred_labels_str = [id2label[px] for px in filtered_pred]

                # on extrait les entités
                pred_ents = extract_entities_bio(pred_labels_str)
                gold_ents = extract_entities_bio(gold_labels_str)

                # on calcule p/r/f1 et on accumule
                pred_set = set(pred_ents)
                gold_set = set(gold_ents)
                tp = len(pred_set & gold_set)
                fp = len(gold_set - pred_set)
                fn = len(gold_set - pred_set)
                tp_all += tp
                fp_all += fp
                fn_all += fn

    prec = tp_all/(tp_all+fp_all) if (tp_all+fp_all)>0 else 0.0
    rec  = tp_all/(tp_all+fn_all) if (tp_all+fn_all)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    model.train()
    return prec, rec, f1

# 4) Entraînement NER
def train_ner(
    pretrained_path,
    train_path,
    dev_path,
    tokenizer,
    label2id,
    id2label,
    num_labels,
    lr=3e-5,
    epochs=3,
    batch_size=16,
    device='cuda'
):
    """
    Fine-tuning NER sur un jeu de données.
    :param pretrained_path: chemin du .pt pour CamembertForPreTraining
    :param train_path: chemin du conll train
    :param dev_path: chemin du conll dev
    :param tokenizer: instance de SentencePieceTokenizer
    :param label2id: dict { "B-PER":5, ...}
    :param id2label: dict inverse {5:"B-PER", ...}
    :param num_labels: nombre total de labels (incluant "O")
    """

    # On charge le modèle pré-entraîné
    pretrained = CamembertForPreTraining.load_pretrained(pretrained_path, device=device)
    # On crée la tête NER
    model = CamembertForNER(pretrained, num_labels=num_labels).to(device)

    # On charge les datasets
    train_data = NERDataset(train_path, tokenizer, label2id)
    dev_data   = NERDataset(dev_path, tokenizer, label2id)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_data, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0.0
    for epoch in range(1, epochs+1):
        # On entraîne
        model.train()
        total_loss=0.0
        for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss/len(train_loader)
        print(f"[Epoch {epoch}] train loss={avg_loss:.4f}")

        # Évaluation F1 sur dev
        prec, rec, f1 = evaluate_ner(model, dev_loader, id2label, device)
        print(f"[Epoch {epoch}] dev prec={prec*100:.2f}, rec={rec*100:.2f}, f1={f1*100:.2f}")

        # On sauvegarde
        if f1>best_f1:
            best_f1=f1
            torch.save(model.state_dict(), "camembert_ner_best.pt")
            print(f"New best model saved (f1={f1*100:.2f})")

    # On peut recharger le best si on veut
    # model.load_state_dict(torch.load("camembert_ner_best.pt"))
    return model