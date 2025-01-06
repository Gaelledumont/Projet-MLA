import os
import glob
import torch
from model.camembert_for_pretraining import CamembertForPreTraining, CamembertConfig
from training.dataset import MLMDataset
from training.trainer import Trainer
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer
from training.utils import set_seed

def main():
    # Vérification CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This training requires GPU.")

    print(f"Found {torch.cuda.device_count()} GPU(s)")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name()}")

    # 1) On fixe la seed pour la reproductibilité
    set_seed(42)

    # 2) Config du modèle
    config = CamembertConfig(
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        masking_strategy="whole_word" # ou "subword"
    )
    model = CamembertForPreTraining(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # 3) On repère les shards
    shard_paths = sorted(glob.glob("data/processed/tokenized_shards_train/shard_*.pt"))
    print(f"Found {len(shard_paths)} shards for pretraining.")

    # Shuffle global des shards pour que l'ordre d'itération soit aléatoire

    # 4) Tokenizer
    tokenizer = SentencePieceTokenizer("data/processed/spm.model")

    # 5) Dataset
    dataset = MLMDataset(
        shards_paths=shard_paths,
        vocab_size=config.vocab_size,
        mask_prob=0.15,
        mask_token_id=tokenizer.mask_token_id(),
        pad_token_id=tokenizer.pad_token_id(),
        max_seq_length=512,
        masking_strategy=config.masking_strategy,
        spm_processor=tokenizer.sp # pour WWM
    )

    # 6) Dataset dev
    dev_shard = glob.glob("data/processed/tokenized_shards_dev/shard_*.pt")
    dev_dataset = None
    if dev_shard:
        dev_dataset = MLMDataset(
            shards_paths=dev_shard,
            vocab_size=config.vocab_size,
            mask_prob=0.15,
            mask_token_id=tokenizer.mask_token_id(),
            pad_token_id=tokenizer.pad_token_id(),
            max_seq_length=512,
            masking_strategy=config.masking_strategy,
            spm_processor=tokenizer.sp
        )
        print(f"Found {len(dev_shard)} dev shards.")
    else:
        print(f"No dev shard found. Perplexity won't be tracked.")

    # 7) Trainer + scheduler polynomial
    total_steps = 100000 # 100k steps mais on peut aller jusqu'à 500k d'après l'article
    warmup_steps = 10000
    trainer = Trainer(
        model=model,
        dataset=dataset,
        batch_size=32,              # on ajuste selon la mémoire
        lr=7e-4,                    # peak lr = 0.0007 comme mentionné dans l'article
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        end_learning_rate=0.0,
        power=1.0,                  # linéaire
        accumulation_steps=256,
        device='cuda',
        checkpoint_steps=10000,
        dev_dataset=dev_dataset,    # dev_dataset si on veut un dev set
        eval_steps=2000,            # toutes les 2000 steps on calcule la perplexité
        use_amp=True
    )

    # On charge un checkpoint si disponible
    checkpoint_dir = "checkpoints"
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)

    start_step = 0
    loss_accum = 0.0
    if latest_checkpoint:
        start_step, loss_accum = trainer.load_checkpoint(latest_checkpoint)

    log_file_path = "training_log.txt"
    with open(log_file_path, "w") as log_file:
        log_file.write("step,loss,lr,val_loss,val_ppl\n")

    # 8) On lance l'entraînement
    trainer.train(start_step, loss_accum, log_file_path) # on va exécuter la boucle tant que step_count < total_steps

    # 9) On sauvegarde
    os.makedirs("ckpts", exist_ok=True)
    torch.save(model.state_dict(), "ckpts/camembert_pretrained_4gb.pt")
    print("Model saved.")

if __name__ == "__main__":
    main()