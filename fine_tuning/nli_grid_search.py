from fine_tuning.nli_trainer import train_nli
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer

def grid_search_nli(
    model_path,
    train_path,
    dev_path,
    tokenizer,
    label2id,
    num_labels,
    device="cuda"
):
    lrs=[1e-5, 2e-5, 3e-5]
    batch_sizes=[16,32]

    best_acc=-1.0
    best_config=None
    for lr in lrs:
        for bs in batch_sizes:
            print(f"\n=== GRID NLI: lr={lr}, bs={bs}, epochs=10 ===")
            dev_acc= train_nli(
                model_path=model_path,
                train_path=train_path,
                dev_path=dev_path,
                tokenizer=tokenizer,
                label2id=label2id,
                num_labels=num_labels,
                epochs=10,
                lr=lr,
                batch_size=bs,
                device=device
            )
            if dev_acc>best_acc:
                best_acc=dev_acc
                best_config=(lr, bs)

    print("\n=====================")
    print(f"BEST ACC on dev= {best_acc*100:.2f}% with config (lr={best_config[0]}, bs={best_config[1]})")

if __name__=="__main__":
    tokenizer=SentencePieceTokenizer("data/processed/spm.model")
    label2id={"entailment":0,"neutral":1,"contradiction":2}
    num_labels= len(label2id)

    grid_search_nli(
        model_path="checkpoints/camembert_pretrained.pt",
        train_path="data/tasks/XNLI-1.0/multinli.train.fr.tsv",
        dev_path="data/tasks/XNLI-1.0/xnli.dev.fr.tsv",
        tokenizer=tokenizer,
        label2id=label2id,
        num_labels=num_labels
    )