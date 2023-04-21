import torch

def set_torch_device(use_gpu=True):
    if not use_gpu:
        return torch.device('cpu')
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device