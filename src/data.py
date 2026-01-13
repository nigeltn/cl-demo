import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

class SplitCIFAR10:
    def __init__(self, root="./data", batch_size=64, debug=False):
        img_transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2023, 0.1994, 0.2010)
            )
        ])
        
        self.train_data = datasets.CIFAR10(
            root=root, train=True, transform=img_transformation, download=True
        )
        self.test_data = datasets.CIFAR10(
            root=root, train=False, transform=img_transformation, download=True
        )
        self.batch_size = batch_size
        self.debug = debug

    def get_task_loader(self, task_labels):
        return (
            self._create_loader(
                self.train_data, task_labels, shuffle=True, debug=self.debug
            ),
            self._create_loader(
                self.test_data, task_labels, shuffle=False, debug=self.debug
            ),
        )

    def _create_loader(self, dataset, filtered_labels, shuffle=False, debug=False):
        targets = torch.tensor(dataset.targets)
        
        mask = torch.zeros(targets.shape, dtype=torch.bool)
        for label in filtered_labels:
            mask |= (targets == label)
            
        indices = mask.nonzero(as_tuple=True)[0]
        
        if debug:
            limit = min(len(indices), 128)
            indices = indices[:limit]
            
        return DataLoader(
            dataset=Subset(dataset, indices=indices),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=2 if not debug else 0,
            pin_memory=True if torch.cuda.is_available() else False
        )