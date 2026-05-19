import logging
import os
from typing import List, Optional

import numpy as np
from torch.utils.data import Dataset


class IndexedNPZForecastingDataset(Dataset):
    """Time-series forecasting dataset backed by data.npz and explicit sample indices.

    This loader follows the ConFormer-style split exactly when an index.npz file is
    provided: each sample stores [input_start, input_end, target_end].
    """

    def __init__(
        self,
        data_file_path: str,
        mode: str,
        input_len: int,
        output_len: int,
        index_file_path: Optional[str] = None,
        train_val_test_ratio: Optional[List[float]] = None,
        data_key: str = "data",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        assert mode in ["train", "valid", "test"], f"Invalid mode: {mode}"
        self.data_file_path = data_file_path
        self.index_file_path = index_file_path
        self.mode = mode
        self.input_len = input_len
        self.output_len = output_len
        self.train_val_test_ratio = train_val_test_ratio or [0.6, 0.2, 0.2]
        self.data_key = data_key
        self.logger = logger

        self.data = self._load_data()
        self.index = self._load_or_build_index()

    def _load_data(self) -> np.ndarray:
        if not os.path.exists(self.data_file_path):
            raise FileNotFoundError(f"data_file_path not found: {self.data_file_path}")
        data = np.load(self.data_file_path)[self.data_key].astype(np.float32, copy=False)
        if data.ndim != 3:
            raise ValueError(f"Expected data with shape [T, N, C], got {data.shape}")
        return data

    def _load_or_build_index(self) -> np.ndarray:
        key = "val" if self.mode == "valid" else self.mode
        if self.index_file_path is not None and os.path.exists(self.index_file_path):
            index_obj = np.load(self.index_file_path)
            index = index_obj[key].astype(np.int64)
            if self.logger is not None:
                self.logger.info(
                    "Loaded %s split from %s with %d samples.",
                    key,
                    self.index_file_path,
                    len(index),
                )
            return index

        num_samples = len(self.data) - self.input_len - self.output_len
        if num_samples <= 0:
            raise ValueError(
                f"Not enough timesteps for input_len={self.input_len}, "
                f"output_len={self.output_len}: data length={len(self.data)}"
            )
        starts = np.arange(num_samples, dtype=np.int64)
        index = np.stack(
            [starts, starts + self.input_len, starts + self.input_len + self.output_len],
            axis=-1,
        )
        train_end = int(self.train_val_test_ratio[0] * len(index))
        val_end = int((self.train_val_test_ratio[0] + self.train_val_test_ratio[1]) * len(index))
        splits = {
            "train": index[:train_end],
            "val": index[train_end:val_end],
            "test": index[val_end:],
        }
        return splits[key]

    def __getitem__(self, index: int) -> dict:
        start, mid, end = self.index[index]
        return {
            "inputs": self.data[start:mid].copy(),
            "target": self.data[mid:end].copy(),
        }

    def __len__(self) -> int:
        return len(self.index)
