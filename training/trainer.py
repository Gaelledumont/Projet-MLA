import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import math
from .schedulers import PolynomialDecayLR

class Trainer:
    def __init__(self, model, dataset, batch_size=32, lr=7e-4, total_steps=100000, warmup_steps=10000, end_learning_rate=0.0, power=1.0, accumulation_steps=256, device='cuda', checkpoint_steps=10000, dev_dataset=None, eval_steps=2000, use_amp=False):
        """
        - total_steps: nombre total de pas d'entraînement
        - warmup_steps: nombre de pas de warmup
        - end_learning_rate: lr min (0.0)
        - accumulation_steps : gradient accumulation
        - power : exponentielle pour polynomial decay (1.0 => linéaire)
        - checkpoint_steps : sauvegarde un checkpoint tous les X steps
        - dev_dataset : un dataset "val" pour calculer une perplexité de validation
        - eval_steps : on fait evaluate_dev() tous les X steps
        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.accumulation_steps = accumulation_steps
        self.checkpoint_steps = checkpoint_steps
        self.dev_dataset = dev_dataset
        self.eval_steps = eval_steps
        self.use_amp = use_amp

        # On calcule steps_per_epoch
        # len(dataset) = total nb d'échantillons
        effective_batch_size = batch_size * accumulation_steps
        dataset_size = len(self.dataset)
        self.steps_per_epoch = math.ceil(dataset_size / effective_batch_size)

        # On en déduit un max d'epochs pour atteindre total_steps
        self.total_epochs = math.ceil(self.total_steps / self.steps_per_epoch)

        print(f"[Trainer] steps per epoch = {self.steps_per_epoch}, total epochs = {self.total_epochs}")

        # DataLoader entraînement
        # on peut shuffle à chaque epoch pour la robustesse
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # Optimizer (Adam avec betas=(0.9, 0.98))
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
            end_learning_rate=self.end_learning_rate,
            power=self.power,
        )

        # On expose l'optimiseur et le scheduler
        self.optimizer = self.optimizer
        self.scheduler = self.scheduler

        # Si on active la Mixed Precision
        if self.use_amp:
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None

        # Si on a un dev_dataset
        if self.dev_dataset is not None:
            self.dev_dataloader = DataLoader(self.dev_dataset, batch_size=batch_size, shuffle=False)
        else:
            self.dev_dataloader = None

        # Vérification CUDA
        if not torch.cuda.is_available() and device == 'cuda':
            raise RuntimeError("CUDA is not available but device='cuda' was specified")

        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        self.train_losses = []
        self.train_steps = []
        self.val_losses = []
        self.val_ppls = []
        self.val_steps = []

    def log_gpu_memory(self):
        if torch.cuda.is_available():
            print(f"GPU memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB allocated, "
                  f"{torch.cuda.memory_reserved() / 1024 ** 2:.2f}MB reserved")

    def train(self, start_step=0, initial_loss_accum=0.0):
        self.model.to(self.device)
        print(f"Model moved to {self.device}")
        print(f"Model parameters device: {next(self.model.parameters()).device}")
        self.log_gpu_memory()

        # On vérifie si la mémoire GPU est utilisée
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

        self.model.train()
        step_count = start_step
        loss_accum = initial_loss_accum

        # Boucle par epoch
        for epoch in range(1, self.total_epochs + 1):
            print(f"===== EPOCH {epoch}/{self.total_epochs} =====")

            # On affiche une barre de progression sur l'epoch
            for batch_idx, (input_ids, attention_mask, labels) in enumerate(tqdm(self.dataloader), start=1):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                # Forward
                if self.use_amp:
                    with autocast(device_type="cuda"):
                        _, loss = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                else:
                    _, loss = self.model(input_ids, attention_mask=attention_mask, labels=labels)

                # Accumulation du gradient
                loss = loss / self.accumulation_steps

                # Backward
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss_accum += loss.item()

                # Mise à jour une fois tous les `accumulation_steps`
                if (step_count + 1) % self.accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()       # mise à jour des poids

                    self.scheduler.step()       # mise à jour du LR
                    self.optimizer.zero_grad()

                step_count += 1

                # Logging
                if step_count % 1000 == 0:
                    avg_loss = loss_accum / 1000
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Step {step_count} | Avg Loss = {avg_loss:.4f} | LR = {current_lr:.6e}")

                    # On stocke la loss et le step
                    self.train_losses.append(avg_loss)
                    self.train_steps.append(step_count)

                    self.log_gpu_memory()
                    loss_accum = 0.0

                    # Validation
                    if self.dev_dataset is not None and (step_count % self.eval_steps == 0):
                        val_loss, val_ppl = self.evaluate_dev()
                        print(f"  [Dev] loss={val_loss:.4f}, ppl={val_ppl:.2f}")

                        # Ajout des valeurs aux listes
                        self.val_losses.append(val_loss)
                        self.val_ppls.append(val_ppl)
                        self.val_steps.append(step_count)

                # Checkpointing
                if step_count % self.checkpoint_steps == 0:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scheduler_state_dict': self.scheduler.state_dict(),
                                'step_count': step_count,
                                'epoch': epoch,
                                'loss_accum': loss_accum
                                }, f"checkpoints/checkpoint_{step_count}.pt")
                    print(f"Checkpoint saved at step {step_count}.")

                # Si on a déjà atteint total_steps, on s'arrête
                if step_count >= self.total_steps:
                    print(f"Reached total steps={self.total_steps}, stopping.")
                    break

            # On boucle sur le dataset "à l'infini" jusqu'au nombre total d'étapes

            if step_count >= self.total_steps:
                break

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