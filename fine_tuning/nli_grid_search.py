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
    # hyperparamètres
    lrs=[1e-5, 2e-5, 3e-5]
    batch_sizes=[16,32]

    best_acc=-1.0
    best_config=None
    best_ckpt=None

    for lr in lrs:
        for bs in batch_sizes:
            ckpt_name=f"nli_lr{lr}_bs{bs}.pt"
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
                device=device,
                out_model_path=ckpt_name
            )
            print(f"[GRID] dev_acc={dev_acc*100:.2f}% => (lr={lr}, bs={bs})")

            if dev_acc>best_acc:
                best_acc=dev_acc
                best_config=(lr, bs)
                best_ckpt = ckpt_name

    print("\n=====================")
    print(f"BEST ACC on dev= {best_acc*100:.2f}% with config (lr={best_config[0]}, bs={best_config[1]})")
    print(f"Checkpoint: {best_ckpt}")

if __name__=="__main__":
    tokenizer=SentencePieceTokenizer("data/processed/spm.model")
    label2id={"entailment":0,"neutral":1,"contradiction":2, "contradictory":3}
    num_labels= len(label2id)

    grid_search_nli(
        model_path="checkpoints/camembert_pretrained_4gb (1).pt",
        train_path="data/tasks/nli/multinli.train.fr.tsv",
        dev_path="data/tasks/nli/xnli.dev.fr.tsv",
        tokenizer=tokenizer,
        label2id=label2id,
        num_labels=num_labels
    )