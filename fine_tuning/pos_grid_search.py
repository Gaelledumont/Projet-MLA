from fine_tuning.pos_trainer import train_pos
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer
import argparse
import yaml
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Grid search for POS tagging fine-tuning')
    parser.add_argument('--pretrained_path', type=str,
                        default="checkpoints/camembert_pretrained_4gb.pt",
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--dataset', type=str,
                        default='gsd',
                        choices=['gsd', 'rhapsodie', 'partut', 'sequoia'],
                        help='Dataset name for configuration')
    parser.add_argument('--train_path', type=str,
                        default=None,  # sera construit automatiquement à partir du dataset
                        help='Path to training data')
    parser.add_argument('--dev_path', type=str,
                        default=None,  # sera construit automatiquement à partir du dataset
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
        args.train_path = f"data/tasks/pos/fr_{args.dataset}-ud-train.conllu"
    if args.dev_path is None:
        args.dev_path = f"data/tasks/pos/fr_{args.dataset}-ud-dev.conllu"

    return args


def load_config(dataset_name: str):
    # On obtient le chemin absolu du répertoire du script
    script_dir = Path(__file__).parent  # donne le chemin vers fine_tuning/
    project_root = script_dir.parent  # remonte au répertoire Projet-MLA/
    config_path = project_root / "configs" / "pos" / f"{dataset_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found for dataset: {dataset_name} at {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)

def grid_search_pos(
        pretrained_path,
        train_path,
        dev_path,
        tokenizer,
        config,
        device="cuda"
):
    # On utilise les paramètres de configuration
    label2id = config['label2id']
    num_labels = len(label2id)
    lrs = config['training']['learning_rates']
    batch_sizes = config['training']['batch_sizes']
    max_epochs = config['training'].get('max_epochs', 30)  # valeur par défaut si non spécifiée

    best_score = -1.0
    best_config = None
    best_ckpt = None

    for lr in lrs:
        for bs in batch_sizes:
            print(f"\n=== GRID: LR={lr}, BS={bs}, max_epochs={max_epochs} ===")
            ckpt_name = f"pos_{config['dataset_name']}_lr{lr}_bs{bs}.pt"
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
                max_epochs=max_epochs,
                out_model_path=ckpt_name
            )
            print(f"[GRID] dev_acc={dev_acc * 100:.2f}% => (lr={lr}, bs={bs})")
            if dev_acc > best_score:
                best_score = dev_acc
                best_config = (lr, bs)
                best_ckpt = ckpt_name

    print("\n==========================")
    print(f"BEST SCORE on dev= {best_score * 100:.2f}% with config (lr={best_config[0]}, bs={best_config[1]})")
    print(f"Checkpoint => {best_ckpt}")

if __name__ == "__main__":
    args = parse_args()

    # On charge la configuration
    config = load_config(args.dataset)

    # On initialise le tokenizer
    tokenizer = SentencePieceTokenizer(args.tokenizer)

    # On lance la recherche sur grille
    grid_search_pos(
        pretrained_path=args.pretrained_path,
        train_path=args.train_path,
        dev_path=args.dev_path,
        tokenizer=tokenizer,
        config=config,
        device=args.device
    )