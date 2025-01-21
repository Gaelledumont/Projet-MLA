# CamemBERT Replication Study

This repository contains a from-scratch implementation of the CamemBERT model in PyTorch, as described in the paper "CamemBERT: a Tasty French Language Model" by Martin et al. (2019). This project was undertaken as a replication study to gain a deeper understanding of the model and to validate the original findings.

## Dependencies

* Language: Python 3.8+
* Framework: PyTorch
* GPU: CUDA / cuDNN installed for hardware acceleration
* NumPy
* sentencepiece
* tqdm
* datasets (for downloading the original OSCAR corpus if needed)

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Data Preparation

1. **Download the 4GB subset of the OSCAR corpus (optional):**
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

2. **Fix encoding errors:**
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

3. **Train the SentencePiece model:**

```bash
python tokenization/train_sentencepiece.py
```

4. **Chunk and tokenize the corpus:**

```bash
python data_preparation/chunk_and_tokenize.py
```

5. **Download the downstream task datasets:**
- **POS tagging and dependency parsing:**
    - Download the Universal Dependencies (UD) French Datasets (GSD, Sequoia, Rhapsodie, ParTUT) in CoNLL-U format.
    - Place the `.conllu` files for each dataset in the `data/tasks/pos/` and `data/tasks/parsing` directories, respectively. You should have `train.conllu`, `dev.conllu` and `test.conllu` for each dataset.
 
- **NER:**
    - Download the WikiNER dataset.
    - **Note:** The WikiNER dataset does not come with a predefined train/dev/test split. We created our own split, using the following code:
      ```python
      import random

      def split_wikiner(input_file, train_file, dev_file, test_file, ratio=(0.8, 0.1, 0.1)):
          with open(input_file, 'r', encoding='utf-8') as f:
          lines = f.readlines()

          # shuffle
          random.shuffle(lines)

          n = len(lines)
          n_train = int(n*ratio[0])
          n_dev   = int(n*ratio[1])
          # test => the remainder
          n_test  = n - n_train - n_dev

          train_part = lines[:n_train]
          dev_part   = lines[n_train:n_train+n_dev]
          test_part  = lines[n_train+n_dev:]

          with open(train_file, 'w', encoding='utf-8') as f:
              for l in train_part:
              f.write(l)
          with open(dev_file, 'w', encoding='utf-8') as f:
              for l in dev_part:
              f.write(l)
          with open(test_file, 'w', encoding='utf-8') as f:
              for l in test_part:
              f.write(l)

          if __name__=="__main__":
              input_file="aij-wikiner-fr-wp3.txt"
              split_wikiner(
                  input_file,
                  train_file="wiki_fr_train.txt",
                  dev_file="wiki_fr_dev.txt",
                  test_file="wiki_fr_test.txt"
              )
              print("Splits created: wiki_fr_train.txt, wiki_fr_dev.txt, wiki_fr_test.txt")
      ```
  - Place the files in `data/tasks/ner`.

- **NLI:**
    - Download the XNLI dataset and extract the French portion.
    - **Note:** The original XNLI dev and test files contain 19 columns. We extracted the relevant columns (premise, hypothesis, label) and saved them using the following code:
      ```python
      with open("xnli.dev.fr.tsv", "r", encoding="utf-8") as fin, \
      open("xnli.dev.fr.3cols.tsv", "w", encoding="utf-8") as fout:
      lines = fin.read().strip().split("\n")
      for line in lines:
          splits=line.split("\t")
          if len(splits)<8:
              continue
          label_str  = splits[2]
          premise    = splits[6]
          hypothesis = splits[7]
          # write
          fout.write(f"{premise}\t{hypothesis}\t{label_str}\n")
      ```
  - Place the files in `data/tasks/nli`.
 
## Pre-training

To pre-train the CamemBERT model, run the following command:
```bash
python training/run_pretraining.py
```

**Important notes:**

- The pre-training script uses a polynomial learning rate decay schedule with a warmup. The default parameters are set in the `Trainer` class in `Projet-MLA/training/trainer.py`
- The script logs the training loss, learning rate, and validation perplexity (if a dev set is provided) every 1000 steps to the console and to a `pretraining_log.txt` file.
- Training will run for 100,000 steps. You can adjust this in `run_pretraining.py`

## Fine-tuning

### POS tagging
```bash
python fine_tuning/pos_grid_search.py \
    --pretrained_path checkpoints/camembert_pretrained_4gb.pt \
    --train_path data/tasks/pos/fr_<dataset>-ud-train.conllu \
    --dev_path data/tasks/pos/fr_<dataset>-ud-dev.conllu \
    --tokenizer data/processed/spm.model
    --device cuda
```
Replace `<dataset>` with `gsd`, `sequoia`, `rhapsodie` or `partut`.

### Dependency parsing
```bash
python fine_tuning/parsing_grid_search.py \
    --pretrained_path checkpoints/camembert_pretrained_4gb.pt \
    --train_path data/tasks/parsing/fr_<dataset>-ud-train.conllu \
    --dev_path data/tasks/parsing/fr_<dataset>-ud-dev.conllu \
    --tokenizer data/processed/spm.model
    --device cuda
```
Replace `<dataset>` with `gsd`, `sequoia`, `rhapsodie` or `partut`.

### NER
```bash
python fine_tuning/ner_grid_search.py \
    --pretrained_path checkpoints/camembert_pretrained_4gb.pt \
    --train_path data/tasks/parsing/fr_<dataset>-ud-train.conllu \
    --dev_path data/tasks/parsing/fr_<dataset>-ud-dev.conllu \
    --tokenizer data/processed/spm.model
    --device cuda
```

### 5. **Compare Results**

- Compare performance with existing CamemBERT
- Analyze the impact of corpus size and origin on results.
