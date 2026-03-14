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
        self,
        data_dir: str = "data/data_panel.csv",
    ):
        super().__init__()
        self.data_dir = data_dir

        self.target_feature = "crime_count"
        self.features = [
            # Common
            "time_sin",
            "time_cos",
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
        self.scale_cols = [
            "centroid_x",
            "centroid_y",
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
        self._validate_required_columns()
        self.data = self.data.sort_values(
            ["cell_id", "crime_group", "time_index"]
        ).reset_index(drop=True)
        self.X = self.data[self.features].values.astype(np.float32)
        self.y = self.data[self.target_feature].values.astype(np.float32)

    def _validate_required_columns(self):
        self.data[self.scale_cols] /= 1e4
        required_columns = {
            "cell_id",
            "crime_group",
            "crime_count",
            "time_index",
            *self.features,
        }
        missing = sorted(required_columns.difference(self.data.columns))
        if missing:
            raise KeyError("Dataset is missing required columns: " + ", ".join(missing))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        X = torch.tensor(self.X[idx])
        y = torch.tensor(self.y[idx])
        return X, y


if __name__ == "__main__":
    dataset = AggDataset()
    record = dataset.__getitem__(101)
    x, y = record
    print(x.shape, y.shape)
    print(len(dataset))
    print(x, y)
