from fine_tuning.pos_trainer import train_pos
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str,
                       default="checkpoints/camembert_pretrained_4gb.pt")
    parser.add_argument('--train_path', type=str,
                       default="data/tasks/pos/fr_gsd-ud-train.conllu")
    parser.add_argument('--dev_path', type=str,
                       default="data/tasks/pos/fr_gsd-ud-dev.conllu")
    parser.add_argument('--tokenizer', type=str,
                       default="data/processed/spm.model")
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def grid_search_pos(
    pretrained_path,
    train_path,
    dev_path,
    tokenizer,
    label2id,
    num_labels,
    device="cuda"
):
    # On définit un grid
    lrs = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4, 3e-4, 5e-4]
    batch_sizes = [16, 32]

    best_score=-1.0
    best_config=None
    best_ckpt=None

    for lr in lrs:
        for bs in batch_sizes:
            print(f"\n=== GRID: LR={lr}, BS={bs}, max_epochs=30 ===")
            ckpt_name=f"pos_lr{lr}_bs{bs}.pt"
            dev_acc = train_pos(
                pretrained_path=pretrained_path,
                train_path=train_path,
                dev_path=dev_path,
                tokenizer=tokenizer,
                label2id=label2id,
                num_labels=num_labels,
                lr=lr,
                batch_size=bs,
                device=device,
                max_epochs=30,
                out_model_path=ckpt_name
            )
            print(f"[GRID] dev_acc={dev_acc*100:.2f}% => (lr={lr}, bs={bs})")
            if dev_acc>best_score:
                best_score=dev_acc
                best_config=(lr,bs)
                best_ckpt=ckpt_name

    print("\n==========================")
    print(f"BEST SCORE on dev= {best_score*100:.2f}% with config (lr={best_config[0]}, bs={best_config[1]})")
    print(f"Checkpoint => {best_ckpt}")

if __name__=="__main__":
    args = parse_args()

    tokenizer = SentencePieceTokenizer(args.tokenizer)
    label2id={
        'ADJ': 0, 'ADP': 1, 'ADV': 2, 'AUX': 3, 'CCONJ': 4, 'DET': 5, 'INTJ': 6, 'NOUN': 7,
        'NUM': 8, 'PRON': 9, 'PROPN': 10, 'PUNCT': 11, 'SCONJ': 12, 'SYM': 13, 'VERB': 14,
        'X': 15, '_': 16
    }
    num_labels = len(label2id)

    grid_search_pos(
        pretrained_path=args.pretrained_path,
        train_path=args.train_path,
        dev_path=args.dev_path,
        tokenizer=tokenizer,
        label2id=label2id,
        num_labels=num_labels,
        device=args.device
    )