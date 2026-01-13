import torch
import torch.nn.functional as F


class EWC:
    def __init__(self, model, dataloader, task_classes, num_samples, device):
        self.model = model
        self.dataloader = dataloader
        self.task_classes = task_classes
        self.num_samples = num_samples
        self.device = device

        self.class_map = {label: idx for idx, label in enumerate(task_classes)}
        self.params = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.fisher = self._compute_fim()

    def _compute_fim(self):
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.model.eval()

        computed_samples = 0
        self.model.zero_grad()

        for imgs, labels in self.dataloader:
            imgs = imgs.to(self.device)

            current_batch_size = labels.shape[0]
            if computed_samples + current_batch_size > self.num_samples:
                limit = self.num_samples - computed_samples
                imgs = imgs[:limit]
                labels = labels[:limit]
                current_batch_size = limit

            for i in range(current_batch_size):
                current_img = imgs[i : i + 1]
                current_label = labels[i : i + 1]
                logits = self.model(current_img)

                masked_logits = logits[:, self.task_classes]
                mapped_labels = torch.tensor(
                    [self.class_map[x.item()] for x in current_label],
                    device=self.device,
                )

                loss = F.cross_entropy(masked_logits, mapped_labels)
                loss.backward()

                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        fisher[n] += p.grad.detach() ** 2

                self.model.zero_grad()

            computed_samples += current_batch_size
            if computed_samples >= self.num_samples:
                break

        for n in fisher:
            fisher[n] /= computed_samples

        return fisher

    def penalty(self, new_model):
        loss = 0.0
        for n, p in new_model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n] * (self.params[n] - p) ** 2).sum()
        return loss
