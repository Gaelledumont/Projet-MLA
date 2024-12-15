# # Pour download OSCAR de façon propre 

# from datasets import load_dataset

# oscar_fr = load_dataset("oscar", "unshuffled_deduplicated_fr")

# # print(oscar_fr)
# # print(oscar_fr["train"].column_names)

# small_oscar = oscar_fr["train"].shuffle(seed=42).select(range(10000)) # seulement 10 000 exemples parce que le dataset complet est trop gros

# with open("oscar_fr_sample.txt", "w") as f:
#     for line in small_oscar["text"]:
#         f.write(line + "\n")

# # sauvegarde dans un fichier texte
        
"""
installer sentencepiece avec sudo apt install sentencepiece  
puis 
spm_train \
  --input=oscar_fr_sample.txt \
  --model_prefix=camembert_sp \
  --vocab_size=32000 \
  --model_type=bpe \
  --character_coverage=1.0
  """

### Tout ça permet d'obtenir les fichiers 'oscar_fr_sample.txt', 'camembert_sp.model' et '.vocab' qui sont sur le github

import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("camembert_sp.model")

text = "Bonjour, comment ça va ?"
encoded = sp.encode(text, out_type=str)  # Tokenize en utilisant des sous-mots
print("Tokens :", encoded)

# Convertir en IDs de vocabulaire (encode)
encoded_ids = sp.encode(text, out_type=int)
print("Token IDs :", encoded_ids)

# Décode
decoded = sp.decode(encoded_ids)
print("Decoded text:", decoded)

"""
from transformers import PreTrainedTokenizerFast

# Créer un tokenizer compatible Hugging Face --> marche pas problème de variable 
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=sp,
    model_max_length=512,
)

encoded = tokenizer("Bonjour, comment ça va ?")
print(encoded)
"""

from transformers import AutoTokenizer

# Avec tokenizer déjà fait (j'arrive pas à en faire moi même) parce qu'il en faut un pour le MLM
tokenizer = AutoTokenizer.from_pretrained("camembert-base")

encoded = tokenizer("Bonjour, comment ça va ?")
print(encoded)

print(tokenizer.mask_token)