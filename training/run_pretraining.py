import os
import glob
import torch
from model.camembert_for_pretraining import CamemForPreTraining, CamembertConfig
from training.dataset import MLMDataset
from training.trainer import Trainer
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer

def main():
    # 1) Config
    config = CamembertConfig(
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        masking_strategy="subword" # ou "whole_word"
    )
    model = CamemForPreTraining(config)

    # 2) On repère les shards
    shard_paths = sorted(glob.glob("date/processed/tokenized_shards/shard_*.pt"))
    print(f"Found {len(shard_paths)} shards.")

    # 3) Dataset
    tokenizer = SentencePieceTokenizer("data/processed/spm.model")
    dataset = MLMDataset(
        shards_paths=shard_paths,
        vocab_size=config.vocab_size,
        mask_prob=0.15,
        mask_token_id=tokenizer.mask_token_id(),
        pad_token_id=tokenizer.pad_token_id(),
        max_seq_length=128,
        masking_strategy=config.masking_strategy,
        spm_processor=tokenizer.sp
    )

    # 4) Trainer
    trainer = Trainer(
        model=model,
        dataset=dataset,
        batch_size=32, # on ajuste selon la mémoire
        lr=1e-4,
        device='cuda'
    )

    # 5) On lance l'entraînement
    total_steps = 100000 # 100k steps mais on peut aller jusqu'à 500k d'après l'article
    trainer.train(total_steps=total_steps)

    # 6) On sauvegarde
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/camembert_pretrained_4gb.pt")
    print("Model saved.")

if __name__ == "__main__":
    main()