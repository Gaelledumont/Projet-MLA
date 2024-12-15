import random

"""WWM is implemented by first randomly sampling 15% of the words in the sequence and then considering all subword tokens in each of this 15% 
for candidate replacement. This amounts to a proportion of selected tokens that is close to the original 15%. These tokens are then either replaced by
<MASK> tokens (80%), left unchanged (10%) or replaced by a random token.
"""

def whole_word_masking(tokenizer, input_ids, mlm_probability=0.15):
    tokens = [tokenizer.sp.IdToPiece(i) for i in input_ids]
    # On détermine les mots : un mot commence par "_"
    words = []
    current_word = []
    for idx, tok in enumerate(tokens):
        if tok.startswith("_"):
            if current_word:
                words.append(current_word)
            current_word = [idx]
        else:
            current_word.append(idx)
    if current_word:
        words.append(current_word)

    num_to_mask = int(round(len(words) * mlm_probability))
    masked_words = random.sample(words, k=num_to_mask)
    labels = [-100]*len(tokens)
    for w in masked_words:
        for idx in w:
            labels[idx] = input_ids[idx]

    # On applique 80/10/10
    for idx, label in enumerate(labels):
        if label != -100:
            p = random.random()
            if p < 0.8:
                input_ids[idx] = tokenizer.sp.IdToPiece("<mask>")
            elif p < 0.9:
                input_ids[idx] = random.randint(0, tokenizer.get_vocab_size()-1)
            else:
                # On garde le même
                pass
    return input_ids, labels

def subword_masking(tokenizer, input_ids, mlm_probability=0.15):
    special_ids = [tokenizer.cls_token_id(), tokenizer.sep_token_id(), tokenizer.pad_token_id(), tokenizer.unk_token_id()]
    labels = [-100]*len(input_ids)
    candidates = [i for i, tid in enumerate(input_ids) if tid not in special_ids]
    num_mask = max(1, int(len(candidates) * mlm_probability))
    masked_indices = random.sample(candidates, num_mask)

    for mi in masked_indices:
        labels[mi] = input_ids[mi]

    for mi in masked_indices:
        p = random.random()
        if p < 0.8:
            input_ids[mi] = tokenizer.sp.IdToPiece("<mask>")
        elif p < 0.9:
            input_ids[mi] = random.randint(0, tokenizer.get_vocab_size()-1)
        else:
            pass
    return input_ids, labels