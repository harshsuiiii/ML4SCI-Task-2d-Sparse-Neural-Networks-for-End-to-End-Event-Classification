import torch
import torch.nn as nn

def compute_effective_flops(model, base_flops):
    nonzero,total=0,0

    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)):
            w = m.weight
            nonzero += torch.count_nonzero(w).item()
            total += w.numel()

    sparsity = 1 - (nonzero/total)
    return base_flops*(1-sparsity)