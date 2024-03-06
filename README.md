# How to use
- Wrap any torch learning rate scheduler to a new learning rate scheduler with learning rate warmup. Make it easy to use when train with pytorch lightning (only need one scheduler)
- Example:
```python
optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=0.0)
scheduler = WrapperWarmupLrScheduler(optimizer, lr_scheduler =torch.optim.lr_scheduler.MultiStepLR, warmup_epochs=5, milestones=[300, 600], gamma=0.1)
```