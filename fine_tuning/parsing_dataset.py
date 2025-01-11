import torch
from torch.utils.data import Dataset

class ParsingDataset(Dataset):
    """
    Lit un fichier .conllu
    Extrait (tokens, heads, relations).
    On ignore les lignes de commentaires '#' et les colonnes qu'on n'utilise pas.

    On stocke un sample = (tokens[], heads[], rels[]).
    """
    def __init__(self, conllu_path, tokenizer, rel2id, max_len=512):
        super().__init__()
        # On parse et on stocke
        self.samples = []
        with open(conllu_path, "r", encoding="utf-8") as f:
            tokens = []
            heads = []
            rels = []
            for line in f:
                line=line.strip()
                # on ignore les commentaires
                if line.startswith("#"):
                    continue
                if not line:
                    # phrase terminée
                    if tokens:
                        self.samples.append((tokens, heads, rels))
                        tokens, heads, rels=[],[],[]
                    continue
                # parse columns
                cols=line.split("\t")
                if len(cols) < 8:
                    # ligne invalide
                    continue
                try:
                    token_id=int(cols[0])
                except:
                    continue
                form = cols[1]
                head = cols[6]
                deprel = cols[7]

                try:
                    head_id=int(head)
                except:
                    head_id=0
                if deprel not in rel2id:
                    # on lève une exception
                    raise ValueError(f"Relation inconnu : {deprel}")

                tokens.append(form)
                heads.append(head_id)
                rels.append(deprel)

            # fin de fichier
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
        input_ids=[2] # <s>
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
        # On stocke sur les positions correspondantes (1er subtoken)
        for i in range(min(tok_count, self.max_len)):
            heads_out[i] = new_heads[i]
            rels_out[i]  = new_rels[i]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attn_mask, dtype=torch.long),
            torch.tensor(heads_out, dtype=torch.long),
            torch.tensor(rels_out,  dtype=torch.long)
        )