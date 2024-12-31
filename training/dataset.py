import torch
from torch.utils.data import Dataset
import random

class MLMDataset(Dataset):
    def __init__(
            self,
            shards_paths,
            vocab_size,
            mask_prob=0.15,
            mask_token_id=4,
            pad_token_id=1,
            max_seq_length=512,
            masking_strategy="whole_word",
            spm_processor=None
    ):
        """
        shards_paths: liste de chemins .pt où sont stochées des listes de séquences tokenisées
        vocab_size: taille vocab
        mask_prob: 0.15 par défaut
        mask_token_id: ID du token <mask>
        pad_token_id: 1 par défaut
        max_seq_length: on tronque ou on pad
        masking_strategy: "subword" ou "whole_word"
        spm_processor: SentencePieceProcessor optionnel pour le WWM
        """
        self.shards_paths = shards_paths
        # On shuffle le tableau de shards
        random.shuffle(self.shards_paths)

        self.vocab_size = vocab_size
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.masking_strategy = masking_strategy
        self.spm_processor = spm_processor

        # Indices
        self.current_shard_index = 0
        self.current_shard_data = None
        self.position_in_shard = 0
        self._load_current_shard()

        # Petit log pour être sûr
        print(f"[MLMDataset] Using mask_token_id={mask_token_id} (vocab_size={vocab_size})")

    def _load_current_shard(self):
        if self.current_shard_index >= len(self.shards_paths):
            self.current_shard_index = 0
            random.shuffle(self.shards_paths)
        shard_path = self.shards_paths[self.current_shard_index]
        self.current_shard_data = torch.load(shard_path)
        random.shuffle(self.current_shard_data) # On mélange un peu
        self.position_in_shard = 0

    def __len__(self):
        # Approx : on considère que la taille effective est la somme des longueurs de chaque shard
        # Mais on peut renvoyer un "grand" nombre ou l'approx
        total = 0
        for p in self.shards_paths:
            # On peut charger juste la taille ou charger le shard
            # On simplifie en comptant la taille brute
            data = torch.load(p, map_location="cpu")
            total += len(data)
        return total

    def __getitem__(self, idx):
        # On ignore idx et on itère "à la volée" sur les shards
        while True:
            if self.position_in_shard >= len(self.current_shard_data):
                # On passe au shard suivant
                self.current_shard_index += 1
                if self.current_shard_index >= len(self.shards_paths):
                    # plus de data
                    self.current_shard_index = 0
                    random.shuffle(self.shards_paths)
                self._load_current_shard()

            seq = self.current_shard_data[self.position_in_shard]
            self.position_in_shard += 1

            # on prépare l'échantillon
            return self._prepare_sample(seq)

    def _prepare_sample(self, token_ids):
        # On tronque ou pad à max_seq_length
        token_ids = token_ids[:self.max_seq_length]
        while len(token_ids) < self.max_seq_length:
            token_ids.append(self.pad_token_id)
        input_ids = torch.tensor(token_ids, dtype=torch.long)

        # On applique le masquage
        if self.masking_strategy == "whole_word":
            input_ids, labels = self.whole_word_mask(input_ids)
        else:
            input_ids, labels = self.subword_mask(input_ids)

        attention_mask = (input_ids != self.pad_token_id).long()
        return input_ids, attention_mask, labels

    def subword_mask(self, input_ids):
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        special_tokens_mask = (input_ids == self.pad_token_id)
        probability_matrix.masked_fill_(special_tokens_mask, 0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100

        # 80%: <mask>
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id

        # 10%: random
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_tokens[indices_random]

        # 10%: unchanged
        return input_ids, labels

    """WWM is implemented by first randomly sampling 15% of the words in the sequence and then considering all subword tokens in each of this 15% 
    for candidate replacement. This amounts to a proportion of selected tokens that is close to the original 15%. These tokens are then either replaced by
    <MASK> tokens (80%), left unchanged (10%) or replaced by a random token.
    """

    def whole_word_mask(self, input_ids):
        """
        Tente d'utiliser is_begin_of_word()
        S'il n'existe pas, fallback vers la méthode reposant sur "_"
        """
        if self.spm_processor and hasattr(self.spm_processor, "is_begin_of_word"):
            return self.wwm_is_begin_of_word(input_ids)
        else:
            return self.wwm_fallback_underscore(input_ids)

    def wwm_is_begin_of_word(self, input_ids):
        """
        Implémentation de WWM utilisant sp.is_begin_of_word(token_id).
        """
        token_list = input_ids.tolist()
        masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)

        # On repère les mots
        word_boundaries = []
        start_idx = 0
        for i, tid in enumerate(token_list):
            if i == 0:
                # Premier token, on le considère comme "continu"
                continue
            if self.spm_processor.is_begin_of_word(tid):
                # on ferme le mot précédent
                word_boundaries.append((start_idx, i-1))
                start_idx = i
        # ferme le dernier
        word_boundaries.append((start_idx, len(token_list) - 1))

        for (wstart, wend) in word_boundaries:
            if random.random() < self.mask_prob:
                for j in range(wstart, wend+1):
                    if token_list[j] != self.pad_token_id:
                        masked_indices[j] = True

        labels = input_ids.clone()
        labels[~masked_indices] = -100

        # On applique le schéma 80/10/10
        #    On note : indices_replaced -> 80% de 'masked_indices' => <mask>
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id

        # 10% random
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_tokens[indices_random]

        # 10% inchangé => rien à faire

        return input_ids, labels

    def wwm_fallback_underscore(self, input_ids):
        """
        Fallback: on détecte les débuts de mots en cherchant l'underscore "▁".
        """
        token_list = input_ids.tolist()
        # On convertit l'ID en string
        if not self.spm_processor:
            return self.subword_mask(input_ids)  # fallback total

        subwords = [self.spm_processor.id_to_piece(tid) for tid in token_list]
        masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)

        # repérage
        word_boundaries = []
        start_idx = 0
        for i, sw in enumerate(subwords):
            if i == 0:
                continue
            if sw.startswith("▁"):
                word_boundaries.append((start_idx, i-1))
                start_idx = i
        word_boundaries.append((start_idx, len(subwords)-1))

        for (wstart, wend) in word_boundaries:
            if random.random() < self.mask_prob:
                for j in range(wstart, wend+1):
                    if token_list[j] != self.pad_token_id:
                        masked_indices[j] = True

        labels = input_ids.clone()
        labels[~masked_indices] = -100

        # 80/10/10
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_tokens[indices_random]

        return input_ids, labels