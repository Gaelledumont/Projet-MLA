# Projet-MLA

# Reproduction du modèle CamemBERT

Ce projet vise à reproduire les résultats expérimentaux de l'article *CamemBERT: a Tasty French Language Model*. Il s'agit d'implémenter en TensorFlow ou PyTorch l'architecture proposée et de valider ses performances sur plusieurs tâches NLP en utilisant un GPU.

---

## 📋 **Objectifs du projet :**
- Implémenter l'architecture CamemBERT (basée sur RoBERTa) pour le français.
- Entraîner le modèle sur des corpus diversifiés (OSCAR).
- Évaluer les performances sur des tâches de NLP : POS tagging, parsing, NER, et NLI.
- Comparer les résultats avec ceux de l'article original.

---

## 🛠️ **Prérequis :**
### **Environnement de développement :**
- **Langages** : Python 3.8+
- **Frameworks** : TensorFlow / PyTorch
- **GPU** : CUDA / cuDNN installés pour l'accélération matérielle

### **Bibliothèques nécessaires :**
```bash
pip install torch torchvision transformers
pip install tensorflow
pip install sentencepiece
```

---

## 📂 **Structure du projet :**
```plaintext
CamemBERT-Reproduction/
│
├── data/                  # Données prétraitées (OSCAR, CCNet, etc.)
├── models/                # Enregistrements des modèles entraînés
├── scripts/               # Scripts d'entraînement et d'évaluation
├── notebooks/             # Analyse des résultats et visualisations
├── report/                # Rapport LaTeX (6-8 pages)
└── README.md              # Documentation du projet
```

---

## 🚀 **Étapes de reproduction :**

### 1. **Prétraitement des données :**
- Télécharger le corpus OSCAR français.
- Nettoyer et tokenizer avec **SentencePiece**.
  
### 2. **Implémentation du modèle :**
- Reproduire l'architecture RoBERTa en utilisant TensorFlow ou PyTorch.
- Implémenter le masquage dynamique des mots entiers (Whole-Word Masking).

### 3. **Entraînement :**
- Configurer l'objectif de **Masked Language Modeling (MLM)**.
- Lancer l'entraînement sur un GPU avec des données de 4 Go et 138 Go :
  ```bash
  python run_pretraining.py --data_path data/oscar_4gb.txt --epochs 10 --batch_size 32
  ```

### 4. **Évaluation sur des tâches aval :**
- Implémenter les tâches POS tagging, NER, parsing, et NLI.
- Utiliser des benchmarks comme **Universal Dependencies (UD)** et **XNLI** :
  ```bash
  python evaluate.py --task pos_tagging --model_path models/camembert_base.pt
  ```

### 5. **Comparaison des résultats :**
- Comparer les performances avec les modèles existants (**mBERT**, **XLM-R**).
- Analyser l'impact de la taille et de l'origine du corpus sur les résultats.

### 6. **Rédaction du rapport LaTeX :**
- Présenter l'architecture, les différences avec l'article, et les résultats obtenus.
- Justifier les choix de conception et discuter les résultats expérimentaux.

### 7. **Création de la vidéo de présentation :**
- Enregistrer une vidéo de 5 minutes expliquant le projet et les résultats.
  - **Tous les participants doivent prendre la parole.**

---

## 📈 **Résultats attendus :**
- Reproduction fidèle des résultats de l'article.
- Validation des performances sur différentes tâches NLP.
- Comparaison critique des performances avec les modèles de référence.

---

## 💡 **Ressources utiles :**
- [Article CamemBERT](https://arxiv.org/abs/1911.03894)
- [Corpus OSCAR](https://oscar-corpus.com/)
- [Documentation Hugging Face](https://huggingface.co/docs)

-------------------------------------------------------------------

# Camembert Implementation

Ce projet implémente une architecture Camembert, un modèle de type Transformer basé sur RoBERTa, spécialement adapté pour le traitement de la langue française. Voici une vue d'ensemble des fichiers et des composants principaux du projet.

## Table des matières
1. [CamembertConfig](#camembertconfig)
2. [Composants du Modèle](#composants-du-modèle)
   - [CamembertEmbeddings](#camembertembeddings)
   - [CamembertSelfAttention](#camembertselfattention)
   - [CamembertEncoder](#camembertencoder)
   - [CamembertModel](#camembertmodel)
3. [Applications](#applications)
   - [CamembertForPreTraining](#camembertforpretraining)
   - [CamembertForTokenClassification](#camembertfortokenclassification)
4. [Initialisation des poids](#initialisation-des-poids)
5. [Notes supplémentaires](#notes-supplémentaires)

---

## CamembertConfig
La classe `CamembertConfig` définit les hyperparamètres du modèle. Voici les principaux attributs :

- **`vocab_size`** : Taille du vocabulaire (par défaut : 32 000).
- **`hidden_size`** : Dimension des représentations cachées.
- **`num_hidden_layers`** : Nombre de couches dans l'encodeur Transformer.
- **`num_attention_heads`** : Nombre de têtes d'attention.
- **`intermediate_size`** : Taille du feed-forward interne dans chaque couche.
- **`hidden_dropout_prob` et `attention_probs_dropout_prob`** : Probabilités de dropout pour éviter le surapprentissage.
- **`max_position_embeddings`** : Longueur maximale des séquences prises en charge.
- **`masking_strategy`** : Stratégie de masquage (par défaut : `whole_word`).

---

## Composants du Modèle

### CamembertEmbeddings
Cette classe gère l'incorporation des mots et des positions dans des vecteurs d'embedding.

- Combine les embeddings de mots (`word_embedding`) et de position (`position_embedding`).
- Applique une normalisation par couches (`LayerNorm`) et un dropout.
- Conserve les informations de position et de sens des tokens.

### CamembertSelfAttention
Implémente le mécanisme d'attention multi-têtes.

- Calcule les matrices **Query**, **Key**, et **Value**.
- Effectue un produit scalaire pour calculer les scores d'attention.
- Applique un dropout pour stabiliser l'entraînement.

### CamembertEncoder
Construit l'encodeur complet en empilant plusieurs couches `CamembertLayer`.

- Chaque couche inclut un module d'attention et un module feed-forward.
- Les représentations sont mises à jour à chaque couche.

### CamembertModel
Structure globale combinant les embeddings et l'encodeur Transformer complet.

- Prépare les tenseurs d'entrée (id des tokens et masque d'attention).
- Passe les données à travers les embeddings et l'encodeur.

---

## Applications

### CamembertForPreTraining
Un modèle pour la pré-formation avec deux composants principaux :

1. **`CamembertModel`** : Base Transformer pour extraire les représentations des séquences.
2. **`lm_head`** : Une tête de prédiction du langage (linear layer) pour générer des logits correspondant aux tokens dans le vocabulaire.

### CamembertForTokenClassification
Modèle pour des tâches comme l'étiquetage de séquences (NER, POS tagging, etc.).

- Utilise le `CamembertModel` comme base.
- Ajoute une couche de classification au-dessus pour prédire les étiquettes des tokens.
- Gère les pertes avec une fonction `CrossEntropyLoss`.

---

## Initialisation des poids

La fonction `roberta_init_weights` initialise les poids du modèle :

- Les poids des couches linéaires et des embeddings sont initialisés avec une distribution normale.
- Les biais sont initialisés à zéro.
- Les poids du vecteur de padding sont réinitialisés à zéro.

---

## Notes supplémentaires

1. **Embeddings** : Les embeddings des mots sont initialisés aléatoirement puis ajustés durant l'entraînement pour capturer le sens des mots et leur contexte.
2. **Attention** : Le mécanisme d'attention aide le modèle à se concentrer sur les parties importantes de la séquence.
3. **Références et ressources** :
   - [Vidéo sur les encodings positionnels](https://www.youtube.com/watch?v=dichIcUZfOw).
   - Article Medium sur l'attention : [Self-Attention Explained](https://medium.com/@geetkal67/attention-networks-a-simple-way-to-understand-self-attention-f5fb363c736d).

Ce README fournit une vue d'ensemble pour naviguer et comprendre le projet. Pour des exemples d'utilisation ou des tests, veuillez vous référer à la documentation ou aux fichiers correspondants.
