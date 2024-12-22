import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import math
from .schedulers import PolynomialDecayLR

class Trainer:
    def __init__(self, model, dataset, batch_size=32, lr=7e-4, total_steps=100000, warmup_steps=10000, end_learning_rate=0.0, power=1.0, accumulation_steps=256, device='cuda', checkpoint_steps=10000, dev_dataset=None):
        """
        - total_steps: nombre total de pas d'entraînement
        - warmup_steps: nombre de pas de warmup
        - end_learning_rate: lr min (0.0)
        - power : exponentielle pour polynomial decay (1.0 => linéaire)
        - checkpoint_steps : sauvegarde un checkpoint tous les X steps
        - dev_dataset : un dataset "val" pour calculer une perplexité de validation
        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.accumulation_steps = accumulation_steps
        self.checkpoint_steps = checkpoint_steps
        self.dev_dataset = dev_dataset

        # DataLoader entraînement
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # Optimizer (Adam avec batas=(0.9, 0.98))
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.98)
        )

        # Scheduler polynomial
        self.scheduler = PolynomialDecayLR(
            optimizer=self.optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            end_learning_rate=end_learning_rate,
            power=power,
        )

        # Si on a un dev_dataset
        if self.dev_dataset is not None:
            self.dev_dataloader = DataLoader(self.dev_dataset, batch_size=batch_size, shuffle=False)
        else:
            self.dev_dataloader = None

    def train(self):
        self.model.to(self.device)
        self.model.train()
        step_count = 0
        loss_accum = 0.0

        while step_count < self.total_steps:
            for input_ids, attention_mask, labels in self.dataloader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                # Forward + Backward
                logits, loss = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = loss / self.accumulation_steps
                loss.backward()
                loss_accum += loss.item()

                # Gradient accumulation
                if (step_count + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()        # mise à jour des poids
                    self.scheduler.step()        # mise à jour du LR
                    self.optimizer.zero_grad()

                step_count += 1

                # Logging
                if step_count % 1000 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(
                        f"Step: {step_count}, "
                        f"Avg Loss last 1000 steps: {loss_accum / 1000:.4f}, "
                        f"LR: {current_lr:.6e}"
                    )
                    loss_accum = 0.0

                    # Validation
                    if self.dev_dataset is not None:
                        val_loss, val_perplexity = self.evaluate_dev()
                        print(f"  [Dev] loss={val_loss:.4f}, ppl={val_perplexity:.2f}")

                # Checkpointing
                if step_count % self.checkpoint_steps == 0:
                    torch.save(self.model.state_dict(), f"checkpoints/checkpoint_{step_count}.pt")
                    print(f"Checkpoint saved at step {step_count}.")

                if step_count >= self.total_steps:
                    break

            # On boucle sur le dataset "à l'infini" jusqu'au nombre total d'étapes

        print(f"Training complete after {step_count} steps.")

    def evaluate_dev(self):
        # Petit calcul de la loss (et perplexité) sur dev
        self.model.eval()
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in self.dev_dataloader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                _, loss = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += loss.item() * input_ids.size(0)
                total_count += input_ids.size(0)

        avg_loss = total_loss / total_count if total_count > 0 else 0.0
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf") # pour éviter overflow
        self.model.train()
        return avg_loss, ppl