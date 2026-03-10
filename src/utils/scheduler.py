import torch.optim as optim

def get_scheduler(optimizer, scheduler_cfg):
    """
    Returns a PyTorch LR scheduler based on config dict.
    Example scheduler_cfg:
      { 'name': 'StepLR', 'step_size': 10, 'gamma': 0.5 }
    """
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_cfg.get('T_max', 1000))
    # name = scheduler_cfg.get('name', 'StepLR')
    # if name == 'StepLR':
    #     return optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_cfg.get('step_size', 10), gamma=scheduler_cfg.get('gamma', 0.5))
    # elif name == 'ReduceLROnPlateau':
    #     return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_cfg.get('mode', 'min'), factor=scheduler_cfg.get('factor', 0.1), patience=scheduler_cfg.get('patience', 10))
    # elif name == 'CosineAnnealingLR':
    #     return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_cfg.get('T_max', 1000))
    # else:
    #     raise ValueError(f"Unknown scheduler: {name}")
