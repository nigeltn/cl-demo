# src/args.py
import argparse
import torch


def get_parser():
    parser = argparse.ArgumentParser(
        description="Continual Learning Benchmark: EWC vs REMIND"
    )

    # --- Global Experiment Settings ---
    parser.add_argument(
        "--strategy",
        type=str,
        default="EWC",
        choices=["Naive", "EWC", "REMIND"],
        help="CL Strategy",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="Where to save JSON logs"
    )

    # NEW: Debug Flag
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (fewer samples, 1 epoch, CPU friendly)",
    )

    # --- Optimization / Training ---
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per task")

    # --- Strategy: EWC Hyperparams ---
    ewc_group = parser.add_argument_group("EWC Configuration")
    ewc_group.add_argument(
        "--ewc_lambda", type=float, default=2000.0, help="Penalty weight for EWC"
    )

    # --- Strategy: REMIND Hyperparams ---
    remind_group = parser.add_argument_group("REMIND Configuration")
    remind_group.add_argument(
        "--remind_buffer_size",
        type=int,
        default=100000,
        help="Max samples in replay buffer",
    )
    remind_group.add_argument(
        "--remind_pq_subspaces",
        type=int,
        default=32,
        help="Number of subspaces for quantization",
    )
    remind_group.add_argument(
        "--remind_pq_centroids",
        type=int,
        default=256,
        help="Centroids per subspace (usually 256)",
    )

    return parser
