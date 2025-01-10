from fine_tuning.ner_trainer import train_ner
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer

tokenizer = SentencePieceTokenizer("data/processed/spm.model")


labels ?????

model = train_ner(
    pretrained_path="camembert_pretrained_4gb.pt",
    train_path="UD_French-FTB/fr_ftb-ud-train.conllu",
    dev_path="UUD_French-FTB/fr_ftb-ud-dev.conllu",
    tokenizer=tokenizer,
    label2id,
    id2label,
    num_labels,
    lr=3e-5,
    epochs=3,
    batch_size=16,
    device='cuda'
)
