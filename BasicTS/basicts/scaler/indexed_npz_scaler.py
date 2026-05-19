import os
from typing import Optional

import numpy as np
import torch

from .base_scaler import BaseScaler


class IndexedNPZStandardScaler(BaseScaler):
    """Z-score scaler fitted on indexed training input windows.

    ConFormer fits a global StandardScaler on x_train[..., 0]. This scaler
    reproduces that behavior for BasicTS runs that use the same index.npz split.
    """

    def __init__(
        self,
        data_file_path: str,
        index_file_path: Optional[str],
        train_ratio: float = 0.6,
        norm_each_channel: bool = False,
        rescale: bool = True,
        data_key: str = "data",
        target_channel: int = 0,
        input_len: int = 12,
        output_len: int = 12,
        chunk_size: int = 2048,
        dataset_name: str = "",
    ) -> None:
        super().__init__(dataset_name, train_ratio, norm_each_channel, rescale)
        self.data_file_path = data_file_path
        self.index_file_path = index_file_path
        self.data_key = data_key
        self.target_channel = target_channel
        self.input_len = input_len
        self.output_len = output_len
        self.chunk_size = chunk_size

        data = np.load(data_file_path)[data_key].astype(np.float32, copy=False)
        train_index = self._load_train_index(data)
        self.mean, self.std = self._fit(data, train_index)
        self.mean = torch.tensor(self.mean, dtype=torch.float32)
        self.std = torch.tensor(self.std, dtype=torch.float32)

    def _load_train_index(self, data: np.ndarray) -> np.ndarray:
        if self.index_file_path is not None and os.path.exists(self.index_file_path):
            return np.load(self.index_file_path)["train"].astype(np.int64)

        num_samples = len(data) - self.input_len - self.output_len
        starts = np.arange(num_samples, dtype=np.int64)
        index = np.stack(
            [starts, starts + self.input_len, starts + self.input_len + self.output_len],
            axis=-1,
        )
        return index[: int(self.train_ratio * len(index))]

    def _fit(self, data: np.ndarray, train_index: np.ndarray):
        if self.norm_each_channel:
            num_nodes = data.shape[1]
            total = np.zeros((1, num_nodes), dtype=np.float64)
            total_sq = np.zeros((1, num_nodes), dtype=np.float64)
            count = 0
            for left in range(0, len(train_index), self.chunk_size):
                chunk_index = train_index[left : left + self.chunk_size]
                chunk = np.stack(
                    [data[start:mid, :, self.target_channel] for start, mid, _ in chunk_index],
                    axis=0,
                ).astype(np.float64, copy=False)
                total += chunk.sum(axis=(0, 1), keepdims=False)[None, :]
                total_sq += np.square(chunk).sum(axis=(0, 1), keepdims=False)[None, :]
                count += chunk.shape[0] * chunk.shape[1]
            mean = total / count
            var = np.maximum(total_sq / count - np.square(mean), 0.0)
            std = np.sqrt(var)
            std[std == 0] = 1.0
            return mean.astype(np.float32), std.astype(np.float32)

        total = 0.0
        total_sq = 0.0
        count = 0
        for left in range(0, len(train_index), self.chunk_size):
            chunk_index = train_index[left : left + self.chunk_size]
            chunk = np.stack(
                [data[start:mid, :, self.target_channel] for start, mid, _ in chunk_index],
                axis=0,
            ).astype(np.float64, copy=False)
            total += float(chunk.sum())
            total_sq += float(np.square(chunk).sum())
            count += chunk.size
        mean = total / count
        var = max(total_sq / count - mean * mean, 0.0)
        std = var ** 0.5
        if std == 0:
            std = 1.0
        return np.float32(mean), np.float32(std)

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        input_data[..., self.target_channel] = (
            input_data[..., self.target_channel] - mean
        ) / std
        return input_data

    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        input_data = input_data.clone()
        input_data[..., self.target_channel] = (
            input_data[..., self.target_channel] * std + mean
        )
        return input_data
