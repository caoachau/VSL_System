"""
data_collection/dataset.py
PyTorch Dataset cho dữ liệu VSL đã qua augment_split.py
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class VSLDataset(Dataset):
    def __init__(self,
                 data_dir:    str,
                 labels_file: str,
                 split:       str = 'train',
                 T:           int = 64):
        """
        Args:
            data_dir   : thư mục data/processed/
            labels_file: path tới vsl_labels.txt
            split      : 'train' | 'val' | 'test'
            T          : số frame (phải khớp với augment_split)
        """
        self.T = T
        self.labels    = open(labels_file, encoding='utf-8').read().splitlines()
        self.label2idx = {l: i for i, l in enumerate(self.labels)}

        self.samples: list[tuple[Path, int]] = []
        split_dir = Path(data_dir) / split
        for label in self.labels:
            label_dir = split_dir / label
            if not label_dir.exists():
                continue
            for npy in sorted(label_dir.glob("*.npy")):
                self.samples.append((npy, self.label2idx[label]))

        if not self.samples:
            raise RuntimeError(
                f"Không tìm thấy dữ liệu trong {split_dir}\n"
                "Hãy chạy augment_split.py trước!"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        seq = np.load(path)          # (T, 75, 3) float32

        # (T, J, C) → (C, T, J) — định dạng cho Conv2d
        x = torch.from_numpy(seq).permute(2, 0, 1).float()
        return x, label

    def class_weights(self) -> torch.Tensor:
        """Tính class weights cho imbalanced dataset"""
        counts = torch.zeros(len(self.labels))
        for _, lbl in self.samples:
            counts[lbl] += 1
        weights = 1.0 / counts.clamp(min=1)
        return weights / weights.sum()
