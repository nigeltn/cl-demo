import torch


def train_one_epoch(
    model, dataloader, criterion, optimizer, device, ewc_list=None, ewc_lambda=0.0
):
    model.train()
    total_loss = 0.0

    for img, labels in dataloader:
        img = img.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        pred = model(img)

        loss = criterion(pred, labels)

        if ewc_list:
            ewc_loss = 0.0
            for ewc_constraint in ewc_list:
                ewc_loss += ewc_constraint.penalty(model)

            loss += (ewc_lambda / 2.0) * ewc_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, task_classes=None):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for img, labels in dataloader:
            img = img.to(device)
            labels = labels.to(device)

            logits = model(img)

            if task_classes is not None:
                task_logits = logits[:, task_classes]

                pred_local = task_logits.argmax(1)

                offset = task_classes[0]
                mapped_labels = labels - offset

                correct += (pred_local == mapped_labels).sum().item()

            else:
                pred = logits.argmax(1)
                correct += (pred == labels).sum().item()

            total += labels.shape[0]

    return correct / total
