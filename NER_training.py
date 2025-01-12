from fine_tuning.ner_trainer import train_ner
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer

tokenizer = SentencePieceTokenizer("data/processed/spm.model")

from datasets import load_dataset

# wikiNER_train = load_dataset("wikiann", "fr", split="train") 
# wikiNER_val = load_dataset("wikiann", "fr", split="validation") 
# wikiNER_test = load_dataset("wikiann", "fr", split="test") 

# wikiNER_train.to_csv("wikiNER_french_train.tsv", sep="\t", index=False)
# wikiNER_val.to_csv("wikiNER_french_val.tsv", sep="\t", index=False)
# wikiNER_test.to_csv("wikiNER_french_test.tsv", sep="\t", index=False)

id2label = {
    "B-LOC": 0,
    "B-ORG": 1,
    "B-PER": 2,
    "I-LOC": 3,
    "I-ORG": 4,
    "I-PER": 5,
    "O": 6
}

label2id = {
    0: "B-LOC",
    1: "B-ORG",
    2: "B-PER",
    3: "I-LOC",
    4: "I-ORG",
    5: "I-PER",
    6: "O"
}


model = train_ner(
    pretrained_path="camembert_pretrained_4gb.pt",
    train_path="wikiNER_french_train.tsv",
    dev_path="wikiNER_french_test.tsv",
    tokenizer=tokenizer,
    label2id=label2id,
    id2label=id2label,
    num_labels=len(label2id),
    lr=3e-5,
    epochs=3,
    batch_size=16,
    device='cpu'
)
