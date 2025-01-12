import torch
from torch.utils.data import Dataset

class ParsingDataset(Dataset):
    """
    Lit un fichier .conllu et mappe chaque token vers un bloc de sous-tokens.
    Extrait (tokens, heads, relations).
    On ignore les lignes de commentaires '#' et les colonnes qu'on n'utilise pas.

    On stocke un sample = (tokens[], heads[], rels[]).
    """
    def __init__(self, conllu_path, tokenizer, rel2id, max_len=512):
        super().__init__()
        # On parse et on stocke
        self.samples = []
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.max_len = max_len

        # 1) On lit le conllu
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
                    # nouvelle phrase
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
                    token_id = int(cols[0])
                except:
                    continue
                form = cols[1]
                head = cols[6]
                deprel = cols[7]
                # conversion head
                try:
                    head_id=int(head)
                except:
                    head_id=0
                if deprel not in self.rel2id:
                    # on lève une exception
                    print(f'Label inconnu {deprel}')
                    continue

                tokens.append(form)
                heads.append(head_id)
                rels.append(deprel)

            # fin de fichier
            if tokens:
                self.samples.append((tokens, heads, rels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, heads, rels = self.samples[idx]
        # on encode les tokens en subwords
        # On va faire la concaténation subwords de chaque token => input_ids
        input_ids=[2] # <s>

        start_positions = [] # pour chaque token i, on stocke la position subtoken de début

        # offset subword actuel
        current_pos = 1 # car on a déjà mis 2 (i.e. <s>)
        # On compte le nombre de tokens effectifs
        # subword alignment minimal : 1er subword => le token
        for i, (tok, hd, rl) in enumerate(zip(tokens,heads,rels)):
            subw = self.tokenizer.encode(tok)
            if len(subw) == 0:
                subw=[self.tokenizer.sp.PieceToId("<unk>")]
            start_positions.append(current_pos)
            input_ids.extend(subw)
            current_pos += len(subw)
            # le 1er subw => correspond au token i

        # On tronque si trop long
        input_ids = input_ids[:self.max_len]
        # On ignore les tokens qui tombent hors subword range
        valid_token_count = len(start_positions)
        while len(input_ids)<self.max_len:
            input_ids.append(self.tokenizer.pad_token_id())
        attention_mask = [1 if x!=self.tokenizer.pad_token_id() else 0 for x in input_ids]

        # new heads
        new_heads = [-1]*valid_token_count
        new_rels = [0]*valid_token_count

        for i in range(valid_token_count):
            hd = heads[i]
            rl = rels[i]

            if hd==0:
                new_heads[i]=0
            else:
                if 1<=hd<=valid_token_count:
                    head_subtoken = start_positions[hd-1]
                    if head_subtoken<self.max_len:
                        new_heads[i] = head_subtoken
                    else:
                        new_heads[i] = -1 # tronqué
                else:
                    new_heads[i] = -1
            new_rels[i] = self.rel2id[rl]

        heads_out = [-1]*self.max_len
        rels_out = [0]*self.max_len

        for i in range(valid_token_count):
            pos = start_positions[i]
            if pos<self.max_len:
                heads_out[pos] = new_heads[i]
                rels_out[pos] = new_rels[i]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(heads_out, dtype=torch.long),
            torch.tensor(rels_out,  dtype=torch.long)
        )