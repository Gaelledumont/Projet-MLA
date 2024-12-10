import torch


"""WWM is implemented by first randomly sampling 15% of the words in the sequence and then considering all subword tokens in each of this 15% 
for candidate replacement. This amounts to a proportion of selected tokens that is close to the original 15%. These tokens are then either replaced by
<MASK> tokens (80%), left unchanged (10%) or replaced by a random token.
"""

def MLM(tokens, tokenizer,  choose_prob=0.15, mask_prob=0.8, random_prob=0.1):
    labels = tokens.clone # On conserve les tokens originaux pour la loss

    # Obtenir les indices des tokens spéciaux : [CLS], [PAD], [SEP], [UNK] et [MASK]
    special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.mask_token_id]

    # Marquer les tokens spéciaux pour qu'ils ne soient pas choisis
    non_maskable_tokens = torch.isin(tokens, torch.tensor(special_tokens, device=tokens.device))

    chosen_tokens = (torch.randn(tokens.shape) < choose_prob) & ~non_maskable_tokens  # On choisit 15% des tokens en ignorant les spéciaux

    random_values = torch.rand(chosen_tokens.shape)  # On assigne une valeur aléatoire pour chaque token

    mask_tokens = random_values < mask_prob  # 80% masqués
    random_tokens = (random_values >= mask_prob) & (random_values < 1 - random_prob)  # 10% aléatoires

    tokens[mask_tokens] = tokenizer.mask_token_id    # On remplace les 80% par des MASK

    random_words = torch.randint(len(tokenizer), tokens.shape, dtype=torch.long)
    tokens[random_tokens] = random_words[random_tokens] # On remplace par tokens aléatoires

    labels[~(mask_tokens | random_tokens)] = -100 # les tokens qui ne sont pas masqués ou changés sont mis à -100 pour être ignorés dans la loss

    return tokens, labels