import torch
import numpy as np
from data_module.dataset import AggDataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Subset, DataLoader


class AggDataLoader:
    def __init__(self, dataset: AggDataset, batch_size: int = 64):
        train_indices, val_indices, test_indices = self._split_data_by_idx(dataset)
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

        weight_sampler = self._get_balanced_train_sampler(dataset)
        train_set = Subset(dataset, train_indices.tolist())
        self.train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=weight_sampler,
        )
        val_set = Subset(dataset, val_indices.tolist())
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_set = Subset(dataset, test_indices.tolist())
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    def _split_data_by_idx(self, dataset: AggDataset):
        df = dataset.data.reset_index(drop=True)

        times = sorted(df["time_index"].unique())
        n_times = len(times)

        train_t = times[int(n_times * 0.6) - 1]
        val_t = times[int(n_times * 0.8) - 1]

        train_indices = np.where(df["time_index"] <= train_t)[0]
        val_indices = np.where(
            (df["time_index"] > train_t) & (df["time_index"] <= val_t)
        )[0]
        test_indices = np.where(df["time_index"] > val_t)[0]
        return train_indices, val_indices, test_indices

    def _get_balanced_train_sampler(self, dataset: AggDataset):
        df = dataset.data.iloc[self.train_indices].reset_index(drop=True)
        labels = (df["crime_count"] > 0).astype(int).to_numpy()

        n_positive = (labels == 1).sum()
        n_negative = (labels == 0).sum()
        pos_weight = torch.tensor(n_negative / n_positive, dtype=torch.float32)
        self.pos_weight = pos_weight

        # Numbers of zero records and nonzero records
        class_counts = np.bincount(labels, minlength=2)
        class_weights = 1 / class_counts
        sample_weights = class_weights[labels]

        weight_sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(self.train_indices),
            replacement=True,
        )
        return weight_sampler


if __name__ == "__main__":
    dataset = AggDataset()
    loader = AggDataLoader(dataset)
    x, y = next(iter(loader.train_loader))
    print(x.shape, y.shape)
