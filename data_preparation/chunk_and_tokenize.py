import os
import sentencepiece as spm
import torch

def chunk_and_tokenize(corpus_path, spm_model, shard_size=100000, output_dir="data/processed/tokenized_shards"):
    """
    Lit le corpus ligne par ligne, tokenize, puis stocke par blocs (shards) de shard_size séquences.
    """
    os.makedirs(output_dir, exist_ok=True)
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model)

    shard_id = 0
    samples = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            tokens = sp.encode_as_ids(line)
            samples.append(tokens)

            if len(samples) >= shard_size:
                torch.save(samples, os.path.join(output_dir, f"shard_{shard_id}.pt"))
                samples = []
                shard_id += 1

    # Save remainder
    if samples:
        torch.save(samples, os.path.join(output_dir, f"shard_{shard_id}.pt"))

    print(f"Tokenized corpus into {shard_size+1} shards in {output_dir}.")

if __name__ == "__main__":
    corpus_path = "data/raw/oscar_fr_4GB_fixed.txt"
    spm_model_path = "data/processed/spm.model"
    chunk_and_tokenize(corpus_path, spm_model_path, shard_size=100000)