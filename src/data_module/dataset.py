import math
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from torch.utils.data import Dataset


class AggDataset(Dataset):
    """
    Features are listed under `__init__()` method.
        Note: `cell_id` and `neighbourhood` are not used during training.
    """

    def __init__(
        self, data_dir: str = "data/data_panel.csv", k: int = 3, mask_ratio: float = 0.3
    ):
        super().__init__()
        self.k = k  # months of time window
        self.mask_ratio = mask_ratio
        self.data_dir = data_dir

        self.target_feature = "crime_count"
        self.features = [
            # Common
            "month_sin",
            "month_cos",
            "time_index",
            "centroid_x",
            "centroid_y",
            # Crime
            "cumulative_crime_count",
            "avg_crime_count",
            "prev_crime_count",
            "crime_group",
            # Light
            "avg_install_month",
            "light_count",
            "total_wattage",
            "total_intensity",
            "avg_wattage",
        ]
        dirs = data_dir.split(".")
        if dirs[-1] == "gpkg":
            self.features.append("geometry")
            df = gpd.read_file(self.data_dir)
        elif dirs[-1] == "parquet":
            df = pd.read_parquet(self.data_dir)
        elif dirs[-1] == "csv":
            df = pd.read_csv(self.data_dir)
        else:
            raise KeyError(f"Unsupported file type: {dirs[-1]}")

        self.data = df
        self.data = self.data.sort_values(
            ["cell_id", "crime_group", "time_index"]
        ).reset_index(drop=True)
        self.X = self.data[self.features].values.astype(np.float32)
        self.y = self.data[self.target_feature].values.astype(np.float32)

        self.window_indices = self._build_window_index()

    def _build_window_index(self):
        window_indices = [np.array([])] * len(self.data)
        for _, group in self.data.groupby(["cell_id", "crime_group"], sort=False):
            diffs = group["time_index"].diff().dropna()
            assert (diffs == 1).all()

            row_ids = group.index.to_numpy(dtype=np.int32)
            for i, row_id in enumerate(row_ids):
                start = max(0, i - self.k)
                hist = row_ids[start:i]
                if len(hist) < self.k:
                    pad = np.full(self.k - len(hist), -1, dtype=np.int32)
                    hist = np.concatenate([pad, hist])

                window_indices[row_id] = hist
        return window_indices

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        """Masks randomly on time-series data"""
        mask = np.zeros(self.k + 1, dtype=bool)
        hist_features = np.zeros((self.k, len(self.features)), dtype=np.float32)

        hist_indices = self.window_indices[idx]
        for i, past_idx in enumerate(hist_indices):
            if past_idx != -1:
                hist_features[i] = self.X[past_idx]
                mask[i] = True
        current = self.X[idx]
        mask[-1] = True

        # Random masking applied for past features
        r = max(0, math.ceil(self.k * self.mask_ratio))
        masked = np.random.choice(self.k, size=r, replace=False)
        hist_features[masked] = 0
        mask[masked] = False

        current = current.reshape(1, -1)
        X = torch.tensor(np.concatenate([hist_features, current], axis=0))
        mask = torch.tensor(mask)
        y = torch.tensor(self.y[idx])
        return X, mask, y


if __name__ == "__main__":
    dataset = AggDataset()
    record = dataset.__getitem__(101)
    x, mask, y = record
    print(x.shape, mask.shape, y.shape)
    print(x, mask, y)
    print(len(dataset))
