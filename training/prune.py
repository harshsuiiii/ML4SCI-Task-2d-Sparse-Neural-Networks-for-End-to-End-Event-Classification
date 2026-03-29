import torch.nn.utils.prune as prune
import torch.nn as nn

def global_prune(model, amount):
    params=[]
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)):
            params.append((m,'weight'))

    prune.global_unstructured(
        params,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )