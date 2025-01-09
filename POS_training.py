from fine_tuning.pos_trainer import train_pos
import sentencepiece as spm

tokenizer = spm.SentencePieceProcessor()
tokenizer.load("data/processed/spm.model")

label2id = {
    "ADJ": 0,
    "ADP": 1,
    "ADV": 2,
    "AUX": 3,
    "CCONJ": 4,
    "DET": 5,
    "INTJ": 6,
    "NOUN": 7,
    "NUM": 8,
    "PART": 9,
    "PRON": 10,
    "PROPN": 11,
    "PUNCT": 12,
    "SCONJ": 13,
    "SYM": 14,
    "VERB": 15,
    "X": 16
}

num_labels = len(label2id)

model = train_pos(
    pretrained_path="camembert_pretrained_4gb.pt",
    train_path="UD_French-GSD/fr_gsd-ud-train.conllu",
    dev_path="UD_French-GSD/fr_gsd-ud-dev.conllu",
    tokenizer=tokenizer,
    label2id=label2id,
    num_labels=num_labels,
    device="cpu"
    )