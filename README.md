# CamemBERT Replication Study

This repository contains a from-scratch implementation of the CamemBERT model in PyTorch, as described in the paper "CamemBERT: a Tasty French Language Model" by Martin et al. (2019). This project was undertaken as a replication study to gain a deeper understanding of the model and to validate the original findings.

---

## 📊 **Project Objectives**

- Implement the CamemBERT architecture (based on RoBERTa) for the French language.
- Train the model on OSCAR.
- Evaluate performance on NLP tasks: POS tagging, dependency parsing, NER and NLI.
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
