import torch
from torch.utils.data import Dataset
from MLM import subword_masking, whole_word_masking

class PretrainingDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=512, mlm_probability=0.15, whole_word_mask=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.whole_word_mask = whole_word_mask

        self.examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if line:
                    input_ids = self.tokenizer.encode(line, max_length=self.max_length, add_special_tokens=True)
                    self.examples.append(input_ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx][:]
        if self.whole_word_mask:
            input_ids, labels = whole_word_masking(self.tokenizer, input_ids, mlm_probability=self.mlm_probability)
        else:
            input_ids, labels = subword_masking(self.tokenizer, input_ids, mlm_probability=self.mlm_probability)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)