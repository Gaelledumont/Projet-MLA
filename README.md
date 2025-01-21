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

a. Download the 4GB subset of the OSCAR corpus (optional):
- If you need to create the `oscar_fr_4GB.txt` file, use the provided Python script in the report to download and extract a 4GB subset from Hugging Face Datasets:
```python
from datasets import load_dataset
import os

# Dataset and configuration name
dataset_name = "oscar-corpus/oscar"
config_name = "unshuffled_deduplicated_fr"

# Destination directory for downloaded data
output_dir = "data/raw"
os.makedirs(output_dir, exist_ok=True)

# Target size (in bytes)
target_size = 4 * 1024 * 1024 * 1024  # 4 GB

# Load the dataset in streaming mode
dataset = load_dataset(dataset_name, config_name, split="train", streaming=True)

# Iterate through the shards and download them
total_size = 0
shard_id = 0
output_file = None

for example in dataset:
    if output_file is None:
        output_filename = os.path.join(output_dir, f"oscar_fr_4GB.txt")
        output_file = open(output_filename, "w", encoding="utf-8")

    output_file.write(example["text"] + "\n")
    total_size += len(example["text"].encode("utf-8"))  # Count the size in bytes

    if total_size >= target_size:
        break

    shard_id += 1

if output_file is not None:
    output_file.close()

print(f"Data downloaded to: {output_filename}")
print(f"Total size: {total_size / (1024 * 1024 * 1024):.2f} GB")
```

b. Fix encoding errors:
- The raw `oscar_fr_4GB.txt` file might contain encoding errors. To fix these, run the following script:
```python
import ftfy

def fix_encoding(input_file, output_file):
    with open(input_file, "r", encoding="utf-8", errors="replace") as f_in, \
            open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            fixed_line = ftfy.fix_text(line)
            f_out.write(fixed_line)

if __name__ == "__main__":
    fix_encoding("data/raw/oscar_fr_4GB.txt", "data/raw/oscar_fr_4GB_fixed.txt")
```

- This script uses the `ftfy` library to automatically correct encoding issues.
- The corrected file will be saved as `oscar_fr_4GB_fixed.txt` in the `data/raw/` directory.

- Tokenize the data using **SentencePiece**.

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
