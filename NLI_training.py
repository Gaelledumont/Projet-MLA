from datasets import load_dataset
from fine_tuning.nli_trainer import train_nli

# xnli_train = load_dataset("xnli", "fr", split="train")
# xnli_val = load_dataset("xnli", "fr", split="validation")

# xnli_train.to_csv("xnli_french_train.tsv", sep="\t", index=False)
# xnli_val.to_csv("xnli_french_val.tsv", sep="\t", index=False)

label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
tokenizer = "data/processed/spm.model"

model = train_nli(
    model_path="camembert_pretrained_4gb.pt",
    train_path="xnli_french_train.tsv",
    dev_path="xnli_french_val.tsv",
    tokenizer=tokenizer,
    label2id=label2id,
    device="cpu"
)
