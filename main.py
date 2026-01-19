# main.py
import torch
import random
import numpy as np
from src.data import SplitCIFAR10
from src.backbone import SplitResNet18
from src.strategies import EWCStrategy, BaseStrategy, REMINDStrategy
from src.logger import ExperimentLogger
from src.args import get_parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main():
    # 1. Config
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    # Overrides for Debug Mode
    if args.debug:
        print("DEBUG MODE ACTIVE: Reducing epochs to 1 and dataset size.")
        args.epochs = 1
        args.batch_size = 32  # Smaller batch for CPU

    print(
        f"\n===Experiment: {args.strategy} | Device: {args.device} | Debug: {args.debug}==="
    )
    logger = ExperimentLogger(args.log_dir, f"cifar10_{args.strategy}_seed{args.seed}")

    # 2. Data & Model
    dataset = SplitCIFAR10(root="./data", batch_size=args.batch_size, debug=args.debug)
    tasks = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    model = SplitResNet18(num_classes=10).to(args.device)

    if args.strategy == "EWC":
        strategy = EWCStrategy(model, args.device, ewc_lambda=args.ewc_lambda)
        strategy.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.strategy == "Naive":
        strategy = BaseStrategy(model, args.device, lr=args.lr)
    elif args.strategy == "REMIND":
        strategy = REMINDStrategy(
            model,
            args.device,
            args.remind_buffer_size,
            args.remind_pq_subspaces,
            args.remind_pq_centroids,
            args.lr,
        )
    else:
        raise ValueError("Unknown strategy")

    # 4. Training Loop
    for task_idx, task_classes in enumerate(tasks):
        print(f"\nTASK {task_idx}: Classes {task_classes}")
        train_loader, _ = dataset.get_task_loader(task_classes)

        # Train
        for epoch in range(args.epochs):
            loss = strategy.train_epoch(train_loader, task_id=task_idx)
            if epoch % 5 == 0:
                print(f"   Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f}")

        # Consolidate
        strategy.on_task_complete(train_loader, task_id=task_idx)

        # Eval
        print(f"\nEvaluation:")
        task_accuracies = []
        for t_id, t_classes in enumerate(tasks[: task_idx + 1]):
            _, test_loader = dataset.get_task_loader(t_classes)
            correct, total = 0, 0
            model.eval()
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(args.device), labels.to(args.device)
                    correct += (model(imgs).argmax(1) == labels).sum().item()
                    total += labels.size(0)

            acc = correct / total if total > 0 else 0
            task_accuracies.append(acc)
            print(f"   Task {t_id} Acc: {acc:.2%}")

        logger.log_task(task_idx, task_accuracies)

    logger.save(args)
    print("\nSaving logging: done.")


if __name__ == "__main__":
    main()
