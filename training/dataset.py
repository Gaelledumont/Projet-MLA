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
            masking_strategy="subword",
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
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.masking_strategy = masking_strategy
        self.spm_processor = spm_processor

        # On garde en mémoire l'index du shard courant
        self.current_shard_index = 0
        self.current_shard_data = None
        # On charge le premier shard
        self._load_current_shard()

    def _load_current_shard(self):
        if self.current_shard_index >= len(self.shards_paths):
            self.current_shard_data = []
            return
        shard_path = self.shards_paths[self.current_shard_index]
        self.current_shard_data = torch.load(shard_path)
        random.shuffle(self.current_shard_data) # On mélange un peu
        self.position_in_shard = 0

    def __len__(self):
        # Approx : on considère que la taille effective est la somme des longueurs de chaque shard
        # Mais on peut renvoyer un "grand" nombre ou l'approx
        total = 0
        for sp in self.shards_paths:
            # On peut charger juste la taille ou charger le shard
            # On simplifie en comptant la taille brute
            data = torch.load(sp, map_location="cpu")
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
                    self._load_current_shard()
                else:
                    self._load_current_shard()

            if len(self.current_shard_data) == 0:
                # plus de data -> fin
                return self._prepare_sample([]) # dummy
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

    def whole_word_mask(self, input_ids):
        """
        Implémentation de Whole-Word Masking (WWM)
        """

        # 1) On convertit input_ids (tensor) en liste python
        token_ids = input_ids.tolist()

        # 2) On détermine le début de chaque mot avec is_begin_of_word
        #    is_begin_of_word(id) retourne True si ce token commence un mot,
        #    False sinon (suite du mot précédent).

        #    On va ainsi collecter des "segments" (word_starts, word_ends)
        word_boundaries = []
        start_idx = 0
        for i, tid in enumerate(token_ids):
            if i == 0:
                # Premier token, on le considère toujours comme un début (sauf si c'est un token spécial)
                continue
            if self.spm_processor.is_begin_of_word(tid):
                # On clôt le mot précédent : [start_idx, i-1]
                word_boundaries.append((start_idx, i-1))
                start_idx = i
        # Fin : on ajoute le dernier mot
        word_boundaries.append((start_idx, len(token_ids) - 1))

        # 3) On prépare un masque booléen 'masked_indices'
        #    qui indiquera quelles positions sont masquées.
        masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)

        # 4) On parcourt chaque "mot" repéré
        for (w_start, w_end) in word_boundaries:
            # On ne masque pas si ce sont des tokens spéciaux (PAD, etc.)
            # Dans la plupart des cas, sp.is_begin_of_word() sera False sur <pad> ou <unk>
            # mais par prudence on va filtrer.
            # Ensuite, on tire p=mask_prob pour masquer ou non ce mot
            if random.random() < self.mask_prob:
                # On masque tous les sous-tokens de [w_start, w_end],
                # sauf ceux qui valent pad_token_id
                for j in range(w_start, w_end + 1):
                    if token_ids[j] != self.pad_token_id:
                        masked_indices[j] = True

        # 5) On crée le vacteur 'labels' = copie de input_ids
        labels = input_ids.clone()
        # Positions non masquées => -100 (pas de contribution à la loss)
        labels[~masked_indices] = -100

        # 6) On applique le schéma 80/10/10
        #    On note : indices_replaced -> 80% de 'masked_indices' => <mask>
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id

        # 10% random
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_tokens[indices_random]

        # 10% inchangé => rien à faire

        return input_ids, labels