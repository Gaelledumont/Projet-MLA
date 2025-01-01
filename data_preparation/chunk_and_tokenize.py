import os
import random
import sentencepiece as spm
import torch

def chunk_and_tokenize(corpus_path, spm_model, shard_size=100000, output_dir="data/processed", dev_ratio=0.1, shuffle_lines=True):
    """
    Lit le corpus ligne par ligne, sépare un dev set, tokenize, puis stocke par blocs (shards).
    :param corpus_path: chemin du fichier texte brut.
    :param spm_model: chemin du .model SentencePiece.
    :param shard_size nombre de lignes tokenisées par shard
    :param output_dir: répertoire où stocker les shards
    :param dev_ratio: proportion de lignes envoyées dans le dev set
    : param shuffle: si True, mélange toutes les lignes avant de scinder train/dev
    """
    # 1) Création des dossiers
    train_shards_dir = os.path.join(output_dir, "tokenized_shards_train")
    dev_shards_dir = os.path.join(output_dir, "tokenized_shards_dev")
    os.makedirs(train_shards_dir, exist_ok=True)
    os.makedirs(dev_shards_dir, exist_ok=True)

    # 2) On charge le tokenizer SentencePiece
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model)

    # 2) On lit toutes les lignes en mémoire
    lines = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line_ = line.strip()
            if line_:
                lines.append(line_)

    print(f"[INFO] Loaded {len(lines)} non-empty lines from '{corpus_path}'")

    # 3) On mélange
    if shuffle_lines:
        random.shuffle(lines)

    # 4) On sépare train/dev
    dev_count = int(len(lines) * dev_ratio)
    dev_lines = lines[:dev_count]
    train_lines = lines[dev_count:]
    print(f"[INFO] Split into {len(train_lines)} lines (train) + {len(dev_lines)} lines (dev).")

    # 5) On tokenize et on stocke en shards - TRAIN
    shard_id = 0
    samples = []
    for i, txt in enumerate(train_lines):
        tokens = sp.encode_as_ids(txt)
        samples.append(tokens)
        if len(samples) >= shard_size:
            shard_path = os.path.join(train_shards_dir, f"shard_{shard_id}.pt")
            torch.save(samples, shard_path)
            samples = []
            shard_id += 1

    # Save remainder
    if samples:
        shard_path = os.path.join(train_shards_dir, f"shard_{shard_id}.pt")
        torch.save(samples, shard_path)

    print(f"[TRAIN] Created {shard_id+1} shards in '{train_shards_dir}'.")

    # 6) On tokenize et stocke en shards - DEV
    if dev_count > 0:
        shard_id_dev = 0
        samples_dev = []
        for i, txt in enumerate(dev_lines):
            tokens = sp.encode_as_ids(txt)
            samples_dev.append(tokens)
            if len(samples_dev) >= shard_size:
                shard_path = os.path.join(dev_shards_dir, f"shard_{shard_id_dev}.pt")
                torch.save(samples_dev, shard_path)
                samples_dev = []
                shard_id_dev += 1
        # remainder dev
        if samples_dev:
            shard_path = os.path.join(dev_shards_dir, f"shard_{shard_id_dev}.pt")
            torch.save(samples_dev, shard_path)

        print(f"[DEV] Created {shard_id_dev+1} shards in '{dev_shards_dir}'.")
    else:
        print(f"[DEV] No dev set created (dev_ratio=0 or no lines).")

if __name__ == "__main__":
    corpus_path = "data/raw/oscar_fr_4GB_fixed.txt"
    spm_model_path = "data/processed/spm.model"

    chunk_and_tokenize(
        corpus_path=corpus_path,
        spm_model=spm_model_path,
        shard_size=100000,
        output_dir="data/processed",
        dev_ratio=0.1,
        shuffle_lines=True
    )