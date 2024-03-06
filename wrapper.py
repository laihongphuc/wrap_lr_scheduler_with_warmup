import warnings
import torch


class WrapperWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, 
                 optimizer,
                 lr_scheduler,
                 warmup_epochs: int = 50,
                 last_epoch: int = -1,
                 verbose: str = 'deprecated',
                 *args, **kwargs):
        self.warmup_epochs = warmup_epochs 
        self.lr_scheduler = lr_scheduler(optimizer, last_epoch = last_epoch, 
                                         verbose = verbose, *args, **kwargs)
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.")
        if self.last_epoch <= self.warmup_epochs:
            scale_factor = min(1., self.last_epoch / self.warmup_epochs)
            return [group['initial_lr'] * scale_factor for group in self.optimizer.param_groups]
        return self.lr_scheduler.get_lr()
    
    def step(self, epoch=None):
        if self.last_epoch <= self.warmup_epochs:
            super().step(epoch)
        else:
            self.lr_scheduler.step(epoch)