from datasets import load_dataset
from fine_tuning.nli_trainer import train_nli
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer

tokenizer = SentencePieceTokenizer("data/processed/spm.model")

# xnli_train = load_dataset("xnli", "fr", split="train")
# xnli_val = load_dataset("xnli", "fr", split="validation")

# xnli_train.to_csv("xnli_french_train.tsv", sep="\t", index=False)
# xnli_val.to_csv("xnli_french_val.tsv", sep="\t", index=False)

label2id = {"0": 0, "1": 1, "2": 2}

model = train_nli(
    model_path="camembert_pretrained_4gb.pt",
    train_path="xnli_french_train.tsv",
    dev_path="xnli_french_val.tsv",
    tokenizer=tokenizer,
    label2id=label2id,
    device="cpu"
)
