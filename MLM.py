import torch

def MLM(tokens, tokenizer,  choose_prob=0.15, mask_prob=0.8, random_prob=0.1):
    labels = tokens.clone # On conserve les tokens originaux pour la loss

    chosen_tokens = torch.randn(tokens.shape) < choose_prob  # On choisit 15% des tokens 

    random_values = torch.rand(chosen_tokens.shape)  # On assigne une valeur aléatoire pour chaque token

    mask_tokens = random_values < mask_prob  # 80% masqués
    random_tokens = (random_values >= mask_prob) & (random_values < 1 - random_prob)  # 10% aléatoires

    tokens[mask_tokens] = tokenizer.mask_token_id    # On remplace les 80% par des MASK

    random_words = torch.randint(len(tokenizer), tokens.shape, dtype=torch.long)
    tokens[random_tokens] = random_words[random_tokens] # On remplace par tokens aléatoires

    return tokens, labels