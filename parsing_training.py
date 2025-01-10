from fine_tuning.parsing_trainer import train_parsing
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer

tokenizer = SentencePieceTokenizer("data/processed/spm.model")

rel2id = {'root': 0, 'acl': 1, 'advcl': 2, 'advmod': 3, 'amod': 4, 'appos': 5, 'aux': 6, 'case': 7, 'cc': 8, 'ccomp': 9, 'clf': 10, 'compound': 11, 'conj': 12, 'cop': 13, 'det': 14, 'discourse': 15, 'expl': 16, 'fixed': 17, 'flat': 18, 'goeswith': 19, 'iobj': 20, 'list': 21, 'mark': 22, 'nmod': 23, 'nsubj': 24, 'nummod': 25, 'obj': 26, 'obl': 27, 'orphan': 28, 'parataxis': 29, 'punct': 30, 'reparandum': 31, 'vocative': 32, 'xcomp': 33, 'aux:pass': 34, 'case:loc': 35, 'cc:preconj': 36, 'det:poss': 37, 'nsubj:pass': 38, 'obl:agent': 39, 'compound:prt': 40, 'aux:tense': 41, 'csubj': 42, 'expl:subj': 43, 'dislocated': 44, 'expl:comp': 45, 'advcl:cleft': 46, 'iobj:agent': 47, 'acl:relcl': 48, 'flat:foreign': 49, 'obl:arg': 50, 'expl:pass': 51, 'obj:lvc': 52, 'aux:caus': 53, 'expl:pv': 54, 'flat:name': 55, 'dep:comp': 56, 'nsubj:caus': 57, 'obl:mod': 58, 'dep': 59, '_': 60, 'obj:agent': 61, 'parataxis:insert': 62, 'nsubj:outer': 63, 'csubj:pass': 64}

model = train_parsing(
    pretrained_path="camembert_pretrained_4gb.pt",
    train_path="UD_French-GSD/fr_gsd-ud-train.conllu",
    dev_path="UD_French-GSD/fr_gsd-ud-dev.conllu",
    tokenizer=tokenizer,
    rel2id=rel2id,
    arc_dim=512,
    rel_dim=512,
    arc_mlp_layers=2,
    rel_mlp_layers=2,
    n_rels=30,
    lr=1e-4,
    epochs=10,
    batch_size=16,
    device='cuda'
)
