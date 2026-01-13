import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.mlp import MLP
from src.ewc import EWC


class Trainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        self.model = MLP(
            input_dim=config.training.input_dim, hidden_dim=config.training.hidden_dim
        ).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.training.lr)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.ewc_list = []  # Stores Fisher/Params from previous tasks

    def get_loader(self, dataset, task_classes, is_train=True):
        loader_data, _ = dataset.get_task_loader(task_classes)
        task_dataset = loader_data.dataset

        return DataLoader(
            task_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=is_train,
            num_workers=2 if self.device.type == "cuda" else 0,
            pin_memory=(self.device.type == "cuda"),
        )

    def train_task(self, task_id, train_loader, task_classes):
        print(f"\n[Task {task_id}] Training on classes {task_classes}...")
        self.model.train()

        offset = task_classes[0]

        for epoch in range(self.config.training.epochs_per_task):
            total_loss = 0.0

            for img, labels in train_loader:
                img, labels = img.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                all_logits = self.model(img)
                task_logits = all_logits[:, task_classes]

                mapped_labels = labels - offset

                loss = self.criterion(task_logits, mapped_labels)

                if self.ewc_list:
                    ewc_loss = 0
                    for ewc_constraint in self.ewc_list:
                        ewc_loss += ewc_constraint.penalty(self.model)

                    loss += (self.config.ewc.lambda_val / 2.0) * ewc_loss

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(
                f"  Epoch {epoch+1}/{self.config.training.epochs_per_task} | Loss: {total_loss / len(train_loader):.4f}"
            )

    def consolidate(self, dataloader, task_classes):
        print(f"  🔒 Consolidating weights for classes {task_classes}...")

        new_ewc = EWC(
            self.model,
            dataloader,
            task_classes,
            self.config.ewc.num_samples,
            self.device,
        )
        self.ewc_list.append(new_ewc)

    def evaluate(self, dataset, tasks):
        self.model.eval()
        accuracies = []

        with torch.no_grad():
            for i, task_classes in enumerate(tasks):
                test_loader = self.get_loader(dataset, task_classes, is_train=False)

                correct = 0
                total = 0
                offset = task_classes[0]

                for img, labels in test_loader:
                    img, labels = img.to(self.device), labels.to(self.device)

                    logits = self.model(img)

                    task_logits = logits[:, task_classes]
                    pred = task_logits.argmax(dim=1)
                    mapped_labels = labels - offset

                    correct += (pred == mapped_labels).sum().item()
                    total += labels.shape[0]

                acc = correct / total if total > 0 else 0.0
                accuracies.append(acc)

        return accuracies
