import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_sparsity(model):
    nonzero,total=0,0
    for m in model.modules():
        if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear)):
            w=m.weight
            nonzero+=torch.count_nonzero(w).item()
            total+=w.numel()
    return 1-(nonzero/total)