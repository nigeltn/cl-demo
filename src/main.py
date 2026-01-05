import argparse
import json
import os
import random
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime

from mlp import MLP
from data import SplitMNIST
from engine import train_one_epoch, evaluate


def get_args():
    parser = argparse.ArgumentParser(description="Continual Learning: Split MNIST")

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="naive_baseline",
        help="Name identifier for the log file.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory where JSON results will be saved.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--epochs_per_task",
        type=int,
        default=5,
        help="Number of epochs to train on each specific task.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=400,
        help="Number of neurons in the hidden layers of the MLP.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["cuda", "cpu"],
        help="Target device.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: uses small subset of data and runs on CPU.",
    )

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

        # Force deterministic algorithms
        # This makes operations slower but 100% reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_json(data, log_dir, filename):
    os.makedirs(log_dir, exist_ok=True)
    filepath = os.path.join(log_dir, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to: {filepath}")


def main():
    args = get_args()
    set_seed(args.seed)

    # Device management
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"

    if args.debug:
        print("💀 DEBUG MODE ENABLED: Using CPU and reduced data/epochs")
        args.device = "cpu"
        args.epochs_per_task = 3

    device = torch.device(args.device)

    print(f"--- Experiment: {args.experiment_name} ---")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    dataset = SplitMNIST(root="./data", batch_size=args.batch_size)
    model = MLP(hidden_dim=args.hidden_dim).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    experiment_log = {"config": vars(args), "results": []}

    for task_id, task_classes in enumerate(tasks):
        print(f"\n[Task {task_id}] Training on classes {task_classes}...")

        train_loader, _ = dataset.get_task_loader(task_classes)

        for epoch in range(args.epochs_per_task):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            print(f"   Epoch {epoch+1}/{args.epochs_per_task} | Loss: {loss:.4f}")

        print(f"   Evaluation after Task {task_id}:")
        task_accuracies = []

        for eval_task_id, eval_classes in enumerate(tasks):
            _, test_loader = dataset.get_task_loader(eval_classes)
            acc = evaluate(model, test_loader, device)
            task_accuracies.append(acc)

            if eval_task_id < task_id:
                state = "PAST"
            elif eval_task_id == task_id:
                state = "CURRENT"
            else:
                state = "FUTURE"

            print(f"      Task {eval_task_id} {eval_classes} [{state}]: {acc:.4f}")

        experiment_log["results"].append(
            {"training_task_id": task_id, "accuracies": task_accuracies}
        )

    if args.debug:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.experiment_name}_{timestamp}.json"
        save_json(experiment_log, args.log_dir, filename)


if __name__ == "__main__":
    main()
