import torch
import sys
from src import config
from src.data import SplitCIFAR10
from src.trainer import Trainer
from src.logger import ExperimentLogger


def main():
    args = config.parse_args()
    cfg = config.load_config(args.config)

    if args.debug:
        cfg.debug = True
        cfg.training.epochs_per_task = 1
        print("💀 DEBUG MODE: 1 Epoch per task")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"--- Experiment: {cfg.experiment_name} ---")
    print(f"Device: {device}")

    logger = ExperimentLogger(cfg.log_dir, cfg.experiment_name)
    dataset = SplitCIFAR10(root="./data", batch_size=cfg.training.batch_size)
    trainer = Trainer(cfg, device)

    # Task Schedule: Split MNIST into 5 tasks of 2 classes each
    tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    # 3. Experiment Loop
    for task_id, task_classes in enumerate(tasks):

        train_loader = trainer.get_loader(dataset, task_classes, is_train=True)
        trainer.train_task(task_id, train_loader, task_classes)
        trainer.consolidate(train_loader, task_classes)

        accuracies = trainer.evaluate(dataset, tasks)
        logger.log_task(task_id, accuracies)

        print(
            f"  📊 Task {task_id} Results: {['{:.2f}'.format(x) for x in accuracies]}"
        )

    logger.save(cfg)


if __name__ == "__main__":
    main()
