import torch


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0

    for img, labels in dataloader:
        img = img.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        pred = model(img)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for img, labels in dataloader:
            img = img.to(device)
            labels = labels.to(device)

            logits = model(img)
            pred = logits.argmax(1)

            correct += (pred == labels).sum().item()
            total += labels.shape[0]

    if correct < 5:
        print(f"   Debug: True Label {labels[0].item()} -> Predicted {pred[0].item()}")
    return correct / total
