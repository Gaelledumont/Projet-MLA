from fine_tuning.parsing_trainer import train_parsing
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer
import argparse
import yaml
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Grid search for dependency parsing fine-tuning')
    parser.add_argument('--pretrained_path', type=str,
                        default="checkpoints/camembert_pretrained_4gb.pt",
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--dataset', type=str,
                        default='gsd',
                        choices=['gsd', 'rhapsodie', 'partut', 'sequoia'],
                        help='Dataset name for configuration')
    parser.add_argument('--train_path', type=str,
                        default=None,
                        help='Path to training data')
    parser.add_argument('--dev_path', type=str,
                        default=None,
                        help='Path to development data')
    parser.add_argument('--tokenizer', type=str,
                        default="data/processed/spm.model",
                        help='Path to tokenizer model')
    parser.add_argument('--device', type=str,
                        default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Construction automatique des chemins si non spécifiés
    if args.train_path is None:
        args.train_path = f"data/tasks/parsing/fr_{args.dataset}-ud-train.conllu"
    if args.dev_path is None:
        args.dev_path = f"data/tasks/parsing/fr_{args.dataset}-ud-dev.conllu"

    return args

def load_config(dataset_name: str):
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config_path = project_root / "configs" / "parsing" / f"{dataset_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found for dataset: {dataset_name} at {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)

def grid_search_parsing(
        pretrained_path,
        train_path,
        dev_path,
        tokenizer,
        config,
        device="cuda"
):
    rel2id = config['rel2id']
    num_labels = len(rel2id)
    lrs = [float(lr) for lr in config['training']['learning_rates']]
    batch_sizes = [int(bs) for bs in config['training']['batch_sizes']]
    max_epochs = int(config['training'].get('max_epochs', 30))

    best_score = -1.0
    best_config = None
    best_ckpt = None

    for lr in lrs:
        for bs in batch_sizes:
            print(f"\n=== GRID: LR={lr}, BS={bs}, max_epochs={max_epochs} ===")
            ckpt_name = f"parsing_{config['dataset_name']}_lr{lr}_bs{bs}.pt"
            dev_las = train_parsing(
                pretrained_path=pretrained_path,
                train_path=train_path,
                dev_path=dev_path,
                tokenizer=tokenizer,
                rel2id=rel2id,
                num_labels=num_labels,
                lr=lr,
                batch_size=bs,
                device=device,
                max_epochs=max_epochs,
                out_model_path=ckpt_name
            )
            print(f"[GRID] dev_las={dev_las * 100:.2f}% => (lr={lr}, bs={bs})")
            if dev_las > best_score:
                best_score = dev_las
                best_config = (lr, bs)
                best_ckpt = ckpt_name

    print("\n==========================")
    print(f"BEST SCORE on dev= {best_score * 100:.2f}% with config (lr={best_config[0]}, bs={best_config[1]})")
    print(f"Checkpoint => {best_ckpt}")

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.dataset)
    tokenizer = SentencePieceTokenizer(args.tokenizer)

    grid_search_parsing(
        pretrained_path=args.pretrained_path,
        train_path=args.train_path,
        dev_path=args.dev_path,
        tokenizer=tokenizer,
        config=config,
        device=args.device
    )