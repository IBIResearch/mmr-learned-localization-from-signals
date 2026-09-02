import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class MMRTrackingDataset(Dataset):
    def __init__(
        self,
        signal_fps: tp.List[tp.Union[str, Path]],
        meta_csv_fp: tp.Union[str, Path],
        signal_postprocess_fn: tp.Optional[tp.Callable] = None,
        gt_postprocess_fn: tp.Optional[tp.Callable] = None,
        coordinate_column_names: tp.Iterable[str] = (
            "position_x",
            "position_y",
            "position_z",
        ),
    ):
        super().__init__()
        self.signals = [np.load(fp, mmap_mode="r") for fp in signal_fps]
        self.signal_start_idx = np.cumsum([0] + [s.shape[0] for s in self.signals])
        meta = pd.read_csv(meta_csv_fp)
        self.gt = meta[list(coordinate_column_names)].values.astype(np.float32)
        self.signal_postprocess_fn = signal_postprocess_fn
        self.gt_postprocess_fn = gt_postprocess_fn

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.signal_start_idx, idx, side="right") - 1
        signal_idx = idx - self.signal_start_idx[file_idx]
        signal = self.signals[file_idx][signal_idx]
        gt = self.gt[idx]
        if self.signal_postprocess_fn is not None:
            signal = self.signal_postprocess_fn(signal)
        if self.gt_postprocess_fn is not None:
            gt = self.gt_postprocess_fn(gt)
        return signal, gt


def normalization(signal):
    return signal / np.max(np.linalg.norm(signal, axis=0))


def noise(signal, noise_level=(0.0, 0.1)):
    sigma = np.random.uniform(*noise_level)
    return signal + np.random.normal(0, sigma, size=signal.shape)


def scale(gt, factor=0.2):
    return gt / factor


class MeasuredNoiseTransform:
    def __init__(self, noise_fp, noise_level=(0.0, 0.1)):
        self.noise = np.load(noise_fp)
        self.noise_level = noise_level

    def __call__(self, signal):
        signal_len = signal.shape[-1]
        noise_frame = np.random.randint(0, self.noise.shape[0])
        noise_start = np.random.randint(0, self.noise.shape[2] - signal_len)
        _sigma = np.random.uniform(*self.noise_level)
        return (
            signal
            + self.noise[noise_frame, :, noise_start : noise_start + signal_len]
            * _sigma
        )
