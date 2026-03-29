import torch
import torch.nn as nn

def train(model, loader, device, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X,y in loader:
            X, y = X.to(device), y.to(device).view(-1,1)

            out = model(X)
            loss = criterion(out,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: {total_loss/len(loader):.4f}")

    return model