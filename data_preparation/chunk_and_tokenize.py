"""
Vue d'ensemble : Ce script est conçu pour préparer un corpus brut afin d'entraîner ou de fine-tuner un modèle de langage 
comme CamemBERT. Il effectue plusieurs opérations essentielles :

1. Lecture d'un fichier texte brut ligne par ligne.
2. Mélange des lignes pour éviter les biais d'ordre dans le corpus.
3. Séparation des données en ensembles d'entraînement (train) et de validation (dev).
4. Tokenisation des lignes en sous-tokens à l'aide de SentencePiece.
5. Découpage des sous-tokens en séquences de longueur fixe (par défaut 512).
6. Sauvegarde des séquences tokenisées en blocs (shards) pour une gestion efficace des données 
   pendant l'entraînement.

Ce script est particulièrement utile pour traiter de grands corpus et les préparer pour des modèles BERT-like.
"""

import os
import random
import sentencepiece as spm
import torch

def chunk_and_tokenize(corpus_path, spm_model, shard_size=100000, output_dir="data/processed", dev_ratio=0.02, shuffle_lines=True, max_seq_len=512):
    """
    Lit le corpus ligne par ligne, sépare un dev set, tokenize en sous-séquences de longueur 512, puis stocke par blocs (shards).

    :param corpus_path: Chemin du fichier texte brut
    :param spm_model: Chemin du .model SentencePiece.
    :param shard_size: Nombre de lignes tokenisées par shard
    :param output_dir: Répertoire où stocker les shards
    :param dev_ratio: Proportion de lignes allouées au dev set
    :param shuffle: si True, mélange toutes les lignes avant de scinder train/dev
    :param max_seq_len: Longueur maximale (en sous-tokens) par séquence
    """
    # 1) Création des dossiers
    train_shards_dir = os.path.join(output_dir, "tokenized_shards_train")
    dev_shards_dir = os.path.join(output_dir, "tokenized_shards_dev")
    os.makedirs(train_shards_dir, exist_ok=True)
    os.makedirs(dev_shards_dir, exist_ok=True)

    # 2) On charge le tokenizer SentencePiece
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model)

    # 3) On lit toutes les lignes en mémoire
    lines = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line_ = line.strip()
            if line_:
                lines.append(line_)

    print(f"[INFO] Loaded {len(lines)} non-empty lines from '{corpus_path}'")

    # 4) On mélange
    if shuffle_lines:
        random.shuffle(lines)

    # 5) On sépare train/dev
    dev_count = int(len(lines) * dev_ratio)
    dev_lines = lines[:dev_count]
    train_lines = lines[dev_count:]
    print(f"[INFO] Split into {len(train_lines)} lines (train) + {len(dev_lines)} lines (dev).")

    # Fonction pour traiter les lignes
    def process_lines_to_shards(lines_list, shard_output_dir):
        """
        Tokenise chaque ligne, segmente en blocs de max_se
