from fine_tuning.ner_trainer import train_ner
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer

def grid_search_ner(
    pretrained_path,
    train_path,
    dev_path,
    tokenizer,
    label2id,
    id2label,
    num_labels,
    device="cuda"
):

    lrs = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4, 3e-4, 5e-4]
    batch_sizes = [16, 32]

    best_score = -1.0
    best_config = None
    best_ckpt=None

    for lr in lrs:
        for bs in batch_sizes:
            ckpt_name=f"ner_lr{lr}_bs{bs}.pt"
            print(f"\n=== GRID NER: lr={lr}, bs={bs}, epochs=10 ===")
            # On appelle train_ner sur 10 epochs
            f1 = train_ner(
                pretrained_path=pretrained_path,
                train_path=train_path,
                dev_path=dev_path,
                tokenizer=tokenizer,
                label2id=label2id,
                id2label=id2label,
                num_labels=num_labels,
                lr=lr,
                epochs=10,
                batch_size=bs,
                device=device,
                out_model_path=ckpt_name
            )
            if f1>best_score:
                best_score=f1
                best_config=(lr,bs)
                best_ckpt = ckpt_name

    print("\n===========================")
    print(f"BEST F1 on dev = {best_score*100:.2f}% with config (lr={best_config[0]}, bs={best_config[1]})")

if __name__=="__main__":
    tokenizer=SentencePieceTokenizer("data/processed/spm.model")
    label2id={'B-LOC': 0, 'B-PER': 1, 'I-LOC': 2, 'I-MISC': 3, 'I-ORG': 4, 'I-PER': 5, 'O': 6,
              'B-MISC': 7, 'B-ORG': 8}
    id2label={v:k for k,v in label2id.items()}
    num_labels=len(label2id)

    grid_search_ner(
        pretrained_path="checkpoints/camembert_pretrained_4gb (1).pt",
        train_path="data/tasks/ner/wiki_fr_train.txt",
        dev_path="data/tasks/ner/wiki_fr_dev.txt",
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label,
        num_labels=num_labels,
        device="cuda"
    )