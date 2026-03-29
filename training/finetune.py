import torch
import torch.nn as nn

def finetune(model, loader, device, epochs=2):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()

    for _ in range(epochs):
        for X,y in loader:
            X, y = X.to(device), y.to(device).view(-1,1)

            out = model(X)
            loss = criterion(out,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()