import torch
from torch.utils.data import DataLoader
import os

class Trainer:
    def __init__(self, model, dataset, optimizer, scheduler, device='cuda', batch_size=32, gradient_accumulation_steps=1, save_steps=10000, output_dir='checkpoints'):
        self.model = model.to(device)
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.global_step = 0

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def train(self, epochs=1):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        for epoch in range(epochs):
            self.model.train()
            for step, batch in enumerate(dataloader):
                input_ids, labels = batch
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                _, loss = self.model(input_ids, labels=labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (step+1)%self.gradient_accumulation_steps==0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    self.global_step += 1

                    if self.global_step%100==0:
                        print(f"Global Step: {self.global_step}, Loss: {loss.item()*self.gradient_accumulation_steps:.4f}")

                    if self.global_step%self.save_steps==0:
                        self.save_checkpoint()

    def save_checkpoint(self):
        ckpt_path = os.path.join(self.output_dir, f"checkpoint-{self.global_step}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step
        }, ckpt_path)
        print(f"Saved checkpoint at {ckpt_path}")