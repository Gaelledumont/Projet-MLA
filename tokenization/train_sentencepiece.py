import sentencepiece as spm
import os

def train_sentencepiece(corpus_path, model_prefix="spm", vocab_size=32000):
    """
    Entraîne un tokenizer SentencePiece sur le fichier corpus_path et génère spm.model et spm.vocab.
    """
    # Paramètres pour un tokenizer de type BPE
    spm_cmd = (
        f"--input={corpus_path} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--character_coverage=0.9995 "
        f"--model_type=bpe"
        f"--unk-id=0 --pad_id=1 --bos_id=2 --eos_id=3 "
        f"--user_defined_symbols=<mask> "
        f"--hard_vocab_limit=false"
        f"--num_threads=8 "
    )
    spm.SentencePieceTrainer.Train(spm_cmd)

if __name__ == "__main__":
    corpus_path = "data/raw/fr_corpus_4GB.txt"
    os.makedirs("data/processed", exist_ok=True)
    train_sentencepiece(corpus_path, model_prefix="data/processed/spm", vocab_size=32000)