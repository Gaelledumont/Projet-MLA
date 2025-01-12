# Projet-MLA

## Reproducing the CamemBERT Model

This project aims to replicate the experimental results of the paper *CamemBERT: a Tasty French Language Model*. The goal is to implement the proposed architecture in TensorFlow or PyTorch and validate its performance on various NLP tasks using a GPU.

---

## 📊 **Project Objectives**

- Implement the CamemBERT architecture (based on RoBERTa) for the French language.
- Train the model on diverse corpora, including OSCAR.
- Evaluate performance on NLP tasks: POS tagging, parsing, NER, and NLI.
- Compare the results with those of the original paper.

---

## 🔧 **Prerequisites**

### **Development Environment**

- **Language**: Python 3.8+
- **Framework**: PyTorch
- **GPU**: CUDA / cuDNN installed for hardware acceleration

### **Required Libraries**

```bash
pip install -r requirements.txt
```

---

## 📂 **Project Structure**

```plaintext
CamemBERT-Reproduction/
🔽
├── data/                  # Preprocessed datasets (OSCAR, CCNet, etc.)
├── models/                # Trained model checkpoints
├── scripts/               # Training and evaluation scripts
├── notebooks/             # Analysis and visualizations
├── report/                # LaTeX report (6-8 pages)
└── README.md              # Project documentation
```

---

## 🚀 **Steps to Reproduce**

### 1. **Data Preparation**

- Download the French OSCAR corpus.
- Clean and tokenize the data using **SentencePiece**.

### 2. **Model Implementation**

- Recreate the CamemBERT architecture using PyTorch.
- Implement dynamic whole-word masking.

### 3. **Training**

- Set up the **Masked Language Modeling (MLM)** objective.
- Train the model on a GPU using both a small sample (4 GB) and a full dataset (138 GB):

```bash
python run_pretraining.py --data_path data/oscar_4gb.txt --epochs 10 --batch_size 32
```

### 4. **Evaluation on Downstream Tasks**

- Implement NLP tasks: POS tagging, NER, parsing, and NLI.
- Use benchmarks like **Universal Dependencies (UD)** and **XNLI**:

```bash
python evaluate.py --task pos_tagging --model_path models/camembert_base.pt
```

### 5. **Compare Results**

- Compare performance with existing models (**mBERT**, **XLM-R**).
- Analyze the impact of corpus size and origin on results.
---

## 💡 **Useful Resources**

- [CamemBERT Paper](https://arxiv.org/abs/1911.03894)
- [OSCAR Corpus](https://oscar-corpus.com/)
- [Hugging Face Documentation](https://huggingface.co/docs)

---

# CamemBERT Implementation Details

This project implements a CamemBERT architecture, a Transformer-based model (similar to RoBERTa) tailored for the French language. Below is a detailed overview of the main components and files in the project.

## Table of Contents

1. [CamemBERTConfig](#camembertconfig)
2. [Model Components](#model-components)
   - [CamemBERTEmbeddings](#camembertembeddings)
   - [CamemBERTSelfAttention](#camembertselfattention)
   - [CamemBERTEncoder](#camembertencoder)
   - [CamemBERTModel](#camembertmodel)
3. [Applications](#applications)
   - [CamemBERTForPreTraining](#camembertforpretraining)
   - [CamemBERTForTokenClassification](#camembertfortokenclassification)
4. [Weight Initialization](#weight-initialization)
5. [Additional Notes](#additional-notes)

---

## CamemBERTConfig

The `CamemBERTConfig` class defines the model's hyperparameters. Key attributes include:

- **`vocab_size`**: Vocabulary size (default: 32,000).
- **`hidden_size`**: Dimension of hidden representations.
- **`num_hidden_layers`**: Number of layers in the Transformer encoder.
- **`num_attention_heads`**: Number of attention heads.
- **`intermediate_size`**: Size of the feed-forward network in each layer.
- **`hidden_dropout_prob` and `attention_probs_dropout_prob`**: Dropout probabilities to prevent overfitting.
- **`max_position_embeddings`**: Maximum sequence length supported.
- **`masking_strategy`**: Masking strategy (default: `whole_word`).

---

## Model Components

### CamemBERTEmbeddings

Manages token and positional embeddings:

- Combines word (`word_embedding`) and positional embeddings (`position_embedding`).
- Applies layer normalization (`LayerNorm`) and dropout.
- Preserves positional and semantic token information.

### CamemBERTSelfAttention

Implements the multi-head attention mechanism:

- Computes **Query**, **Key**, and **Value** matrices.
- Calculates attention scores via scaled dot-product.
- Applies dropout for stability.

### CamemBERTEncoder

Builds the complete encoder by stacking multiple `CamemBERTLayer` instances:

- Each layer includes an attention module and a feed-forward network.
- Updates token representations iteratively across layers.

### CamemBERTModel

Combines embeddings and the full Transformer encoder:

- Prepares input tensors (token IDs and attention masks).
- Passes data through embeddings and the encoder.

---

## Applications

### CamemBERTForPreTraining

A model designed for pre-training with two main components:

1. **`CamemBERTModel`**: Base Transformer for extracting sequence representations.
2. **`lm_head`**: A prediction head (linear layer) for generating logits for vocabulary tokens.

### CamemBERTForTokenClassification

Specialized for sequence labeling tasks (e.g., NER, POS tagging):

- Uses `CamemBERTModel` as the base.
- Adds a classification layer for token label prediction.
- Computes loss with `CrossEntropyLoss`.

---

## Weight Initialization

The `roberta_init_weights` function initializes model weights:

- Linear layer and embedding weights are initialized with a normal distribution.
- Biases are set to zero.
- Padding vector weights are reset to zero.

---

## Additional Notes

1. **Embeddings**: Word embeddings are randomly initialized and fine-tuned during training to capture word meaning and context.
2. **Attention**: The attention mechanism helps the model focus on important parts of the sequence.
