import torch
from sklearn.metrics import roc_auc_score

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for X,y in loader:
            X = X.to(device)
            out = torch.sigmoid(model(X)).cpu()

            preds.append(out)
            labels.append(y.view(-1,1))

    preds = torch.cat(preds)
    labels = torch.cat(labels)

    acc = ((preds>0.5)==labels).float().mean().item()
    auc = roc_auc_score(labels.numpy(), preds.numpy())

    return acc, auc