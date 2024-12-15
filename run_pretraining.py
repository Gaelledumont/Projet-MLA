import yaml
import torch
import torch.optim as optim
from model.config import TransformerConfig
from model.CamemBERTForMaskedLM import CamemBERTForMaskedLM
from data.tokenizer import CamemBertTokenizer
from data.dataset import PretrainingDataset
from training.trainer import Trainer
from training.schedulers import PolynomialDecayLR

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Récupération des configs
    num_epochs = config["train"]["tune"]["num_epochs"]
    lr = config["train"]["tune"]["lr"]
    warmup_steps = config["train"]["tune"]["warmup_steps"]
    total_steps = config["train"]["tune"]["total_steps"]

    vocab_size = config["model"]["vocab_size"]
    max_len = config["model"]["max_len"]
    hidden_dim = config["model"]["hidden_dim"]
    num_heads = config["model"]["num_heads"]
    num_layers = config["model"]["layers"]
    dropout = config["model"]["dropout"]
    intermediate_size = config["model"]["intermediate_size"] if "intermediate_size" in config["model"] else 3072
    pad_token_id = config["model"]["pad_token_id"]
    whole_word_mask = config["model"]["whole_word_mask"]

    train_file = config["data"]["train_file"]
    sp_model_path = config["data"]["sp_model_path"]
    mlm_probability = config["data"]["mlm_probability"]

    batch_size = config["training"]["batch_size"]
    gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
    save_steps = config["training"]["save_steps"]
    output_dir = config["training"]["output_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # On initialise le tokenizer
    tokenizer = CamemBertTokenizer(sp_model_path)

    # On construit la config du modèle
    model_config = TransformerConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size,
        dropout_prob=dropout,
        max_position_embeddings=max_len+2, # CamemBERT a 514 pour 512 tokens, incluant <s> et </s>
        pad_token_id=pad_token_id
    )

    # On charge le dataset
    dataset = PretrainingDataset(tokenizer, train_file, max_length=max_len, mlm_probability=mlm_probability, whole_word_mask=whole_word_mask)

    # On initialise le modèle
    model = CamemBERTForMaskedLM(model_config)

    # Optimizer Adam avec betas (0.9,0.98), weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.98), eps=1e-6, weight_decay=0.01)

    # Scheduler polynomial decay
    scheduler = PolynomialDecayLR(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    trainer = Trainer(model, dataset, optimizer, scheduler, device=device, batch_size=batch_size,
                      gradient_accumulation_steps=gradient_accumulation_steps, save_steps=save_steps, output_dir=output_dir)
    trainer.train(epochs=num_epochs)
