# Projet-MLA

## Reproducing the CamemBERT Model

This project aims to replicate the experimental results of the paper *CamemBERT: a Tasty French Language Model*. The goal is to implement the proposed architecture in TensorFlow or PyTorch and validate its performance on various NLP tasks using a GPU.

---

## 📊 **Project Objectives**

- Implement the CamemBERT architecture (based on RoBERTa) for the French language.
- Train the model on OSCAR.
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

## 🚀 **Steps to Reproduce**

### 1. **Data Preparation**

- Download the French OSCAR corpus.
- Clean and tokenize the data using **SentencePiece**.

### 2. **Model Implementation**

- Recreate the CamemBERT architecture using PyTorch.
- Implement dynamic whole-word masking.

### 3. **Training**

- Set up the **Masked Language Modeling (MLM)** objective.
- Train the model on a GPU using a small sample (4 GB)

### 4. **Evaluation on Downstream Tasks**

- Implement NLP tasks: POS tagging, NER, parsing, and NLI.
- Using benchmarks like **Universal Dependencies (UD)** and **XNLI**:

### 5. **Compare Results**

- Compare performance with existing CamemBERT
- Analyze the impact of corpus size and origin on results.
---
