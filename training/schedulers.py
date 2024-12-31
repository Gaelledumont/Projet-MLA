class PolynomialDecayLR:
    def __init__(self, optimizer, warmup_steps, total_steps, end_learning_rate=0.0, power=1.0, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_epoch = last_epoch
        self.current_step = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        self.current_step += 1
        for i, group in enumerate(self.optimizer.param_groups):
            lr = self.get_lr(i)
            group['lr'] = lr

    def get_lr(self, param_group_idx):
        if self.current_step < self.warmup_steps:
            return self.base_lrs[param_group_idx] * self.current_step / self.warmup_steps
        else:
            # progress dans [0,1] après le warmup
            progress = (self.current_step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
            progress = min(max(progress, 0.0), 1.0)  # clamp
            decay = (1 - progress)**self.power
            return (self.base_lrs[param_group_idx] - self.end_learning_rate) * decay + self.end_learning_rate