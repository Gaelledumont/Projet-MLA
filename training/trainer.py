import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import tqdm as tqdm

class Trainer:
    def __init__(self, model, dataset, batch_size=16, lr=1e-4, max_seq_length=128, accumulation_steps=1, device='cuda'):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.accumulation_steps = accumulation_steps

    def train(self, total_steps=100000):
        self.model.to(self.device)
        self.model.train()
        step_count = 0
        loss_accum = 0.0

        while step_count < total_steps:
            for input_ids, attention_mask, labels in self.dataloader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                logits, loss = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = loss / self.accumulation_steps
                loss.backward()
                loss_accum += loss.item()

                if (step_count + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                step_count += 1
                if step_count % 1000 == 0:
                    print(f"Step: {step_count}, loss: {loss_accum / 1000:.4f}")
                    loss_accum = 0.0

                if step_count >= total_steps:
                    break

            # On boucle sur le dataset "à l'infini" jusqu'au nombre total d'étapes

        print("Training complete.")