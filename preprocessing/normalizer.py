"""
preprocessing/normalizer.py
Normalize skeleton sequence trước khi đưa vào model.
- Center theo hip center (joints 23, 24)
- Scale theo shoulder width
- Pad hoặc crop về T frames
"""
import numpy as np


def normalize_sequence(seq: np.ndarray, T: int = 64) -> np.ndarray:
    """
    Args:
        seq : (N, 75, 3) — N frames thực tế
        T   : số frame chuẩn (default 64)
    Returns:
        (T, 75, 3) normalized float32
    """
    seq = _pad_or_crop(seq, T)
    seq = _center_and_scale(seq)
    return seq.astype(np.float32)


def _pad_or_crop(seq: np.ndarray, T: int) -> np.ndarray:
    N = len(seq)
    if N == T:
        return seq
    if N > T:
        # Crop từ giữa
        start = (N - T) // 2
        return seq[start: start + T]
    # Pad cuối bằng frame cuối
    pad = np.repeat(seq[-1:], T - N, axis=0)
    return np.concatenate([seq, pad], axis=0)


def _center_and_scale(seq: np.ndarray) -> np.ndarray:
    """
    Center: lấy trung điểm hip (joint 23 & 24) làm gốc tọa độ
    Scale : shoulder width (joint 11 – 12) = 1 đơn vị
    """
    seq = seq.copy()

    # Center xy theo hip
    hip_center = (seq[:, 23, :2] + seq[:, 24, :2]) / 2  # (T, 2)
    seq[:, :, :2] -= hip_center[:, np.newaxis, :]

    # Scale
    shoulder_w = np.linalg.norm(
        seq[:, 11, :2] - seq[:, 12, :2], axis=1  # (T,)
    ).mean()
    if shoulder_w > 1e-6:
        seq /= shoulder_w

    return seq
