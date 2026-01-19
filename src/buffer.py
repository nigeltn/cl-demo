import torch
import numpy as np
import faiss


class RemindBuffer:
    def __init__(
        self, buffer_size, feature_dim, pq_subspaces=32, pq_centroids=256, device="cpu"
    ):
        self.max_size = buffer_size
        self.current_size = 0
        self.device = device

        if feature_dim % pq_subspaces != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by pq_subspaces ({pq_subspaces})"
            )

        self.pq_subspaces = pq_subspaces
        self.code_bits = int(np.log2(pq_centroids))  # 2^(code_bits) = num_centroids

        self.pq = faiss.IndexPQ(feature_dim, pq_subspaces, self.code_bits)
        self.is_trained = False

        self.code_size = self.pq.sa_code_size()
        self.codes = np.zeros((buffer_size, self.code_size), dtype=np.uint8)
        self.labels = np.zeros(buffer_size, dtype=np.int64)

    def train_quantizer(self, features):
        if self.is_trained:
            return

        print(
            f"[REMIND Quantization] Training PQ Quantizer on {len(features)} samples..."
        )

        data_np = features.detach().cpu().numpy().astype("float32")
        n = data_np.shape[0]
        self.pq.train(data_np)
        self.is_trained = True

        print(f"[REMIND Quantization] PQ Quantizer has been successfully trained.")

    def add_data(self, features, labels):
        if not self.is_trained:
            raise RuntimeError("PQ must be trained before adding data!")
        space_left = self.max_size - self.current_size

        if space_left == 0:
            return

        n = features.shape[0]
        data_np = features.detach().cpu().numpy().astype("float32")

        new_codes = self.pq.sa_encode(data_np)

        if n > space_left:
            n = space_left
            new_codes = new_codes[:n]
            labels = labels[:n]

        start = self.current_size
        end = self.current_size + n

        self.codes[start:end] = new_codes
        self.labels[start:end] = labels.detach().cpu().numpy()
        self.current_size += n

    def get_batch(self, batch_size):
        if self.current_size == 0:
            return None, None

        indices = np.random.choice(self.current_size, batch_size, replace=False)
        batch_codes = self.codes[indices]
        batch_labels = self.labels[indices]

        decoded_features = self.pq.sa_decode(batch_codes)

        return (
            torch.tensor(decoded_features, dtype=torch.float32).to(self.device),
            torch.tensor(batch_labels, dtype=torch.long).to(self.device),
        )
