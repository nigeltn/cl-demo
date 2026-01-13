import json
import os
from datetime import datetime
from types import SimpleNamespace


class ExperimentLogger:
    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.results = []
        os.makedirs(log_dir, exist_ok=True)

    def log_task(self, task_id, accuracies):
        """Log the results after a specific task."""
        self.results.append({"training_task_id": task_id, "accuracies": accuracies})

    def _to_dict(self, obj):
        """Recursively convert SimpleNamespace to dict."""
        if isinstance(obj, SimpleNamespace):
            return {k: self._to_dict(v) for k, v in vars(obj).items()}
        elif isinstance(obj, dict):
            return {k: self._to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_dict(v) for v in obj]
        else:
            return obj

    def save(self, config_obj):
        """Save the full experiment log to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment_name}_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)

        # Convert config object (SimpleNamespace) to a clean dict
        config_dict = self._to_dict(config_obj)

        full_log = {"config": config_dict, "results": self.results}

        with open(filepath, "w") as f:
            json.dump(full_log, f, indent=4)
        print(f"✅ Results saved to: {filepath}")
