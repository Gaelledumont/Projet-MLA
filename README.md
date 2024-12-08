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
  python train.py --data_path data/oscar_4gb.txt --epochs 10 --batch_size 32
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
