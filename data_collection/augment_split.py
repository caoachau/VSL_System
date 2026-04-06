"""
data_collection/augment_split.py
Augment dữ liệu thô và chia train/val/test.

Augmentations:
  1. Gốc (không thay đổi)
  2. Flip ngang (mirror tay)
  3. Gaussian noise
  4. Tốc độ chậm x0.8
  5. Tốc độ nhanh x1.2

Split: 70% train / 15% val / 15% test (theo từng label)

Chạy: python data_collection/augment_split.py
"""
import numpy as np
from pathlib import Path
import random
import shutil
import sys, os
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.normalizer import normalize_sequence

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Augmentation functions ────────────────────────────────────────────

def aug_flip(seq: np.ndarray) -> np.ndarray:
    """Flip x (tay trái ↔ phải)"""
    s = seq.copy()
    s[:, :, 0] = 1.0 - s[:, :, 0]
    # Hoán đổi left hand (33-53) và right hand (54-74)
    tmp = s[:, 33:54, :].copy()
    s[:, 33:54, :] = s[:, 54:75, :]
    s[:, 54:75, :] = tmp
    return s

def aug_noise(seq: np.ndarray, std: float = 0.005) -> np.ndarray:
    """Thêm Gaussian noise nhỏ"""
    return (seq + np.random.normal(0, std, seq.shape)).clip(0, 1).astype(np.float32)

def aug_speed(seq: np.ndarray, factor: float = 0.8, T: int = 64) -> np.ndarray:
    """Thay đổi tốc độ rồi pad/crop về T"""
    N = len(seq)
    new_N = max(1, int(N * factor))
    indices = np.linspace(0, N - 1, new_N).astype(int)
    sped = seq[indices]
    if len(sped) < T:
        pad = np.repeat(sped[-1:], T - len(sped), axis=0)
        sped = np.concatenate([sped, pad])
    return sped[:T]

def augment_all(seq: np.ndarray, T: int = 64) -> list[np.ndarray]:
    """Trả về 5 biến thể của 1 clip"""
    seq = normalize_sequence(seq, T)
    return [
        seq,
        aug_flip(seq),
        aug_noise(seq),
        aug_speed(seq, factor=0.8,  T=T),
        aug_speed(seq, factor=1.25, T=T),
    ]

AUGMENT_NAMES = ['orig', 'flip', 'noise', 'slow', 'fast']

# ── Main ──────────────────────────────────────────────────────────────

def process(raw_dir: str  = "data/raw",
            proc_dir: str = "data/processed",
            T: int        = 64,
            train_ratio: float = 0.70,
            val_ratio:   float = 0.15):

    raw_path  = Path(raw_dir)
    proc_path = Path(proc_dir)
    labels    = sorted([d.name for d in raw_path.iterdir() if d.is_dir()])

    if not labels:
        print(f"✗ Không tìm thấy dữ liệu trong {raw_dir}")
        print("  Hãy chạy collector.py trước!")
        return

    print(f"Tìm thấy {len(labels)} nhãn: {labels}")
    print(f"Split: {train_ratio:.0%} train / {val_ratio:.0%} val / {1-train_ratio-val_ratio:.0%} test\n")

    for split in ['train', 'val', 'test']:
        (proc_path / split).mkdir(parents=True, exist_ok=True)

    total_train = total_val = total_test = 0

    for label in labels:
        clips = sorted((raw_path / label).glob("*.npy"))
        if not clips:
            print(f"  [skip] '{label}' — không có clip nào")
            continue

        random.shuffle(clips)
        n = len(clips)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        split_map = {
            'train': clips[:n_train],
            'val':   clips[n_train: n_train + n_val],
            'test':  clips[n_train + n_val:],
        }

        for split, files in split_map.items():
            out_dir = proc_path / split / label
            out_dir.mkdir(parents=True, exist_ok=True)

            for fp in tqdm(files, desc=f"  {label}/{split}", leave=False):
                seq = np.load(fp)

                if split == 'train':
                    for aug_seq, aug_name in zip(augment_all(seq, T), AUGMENT_NAMES):
                        np.save(out_dir / f"{fp.stem}_{aug_name}.npy", aug_seq)
                else:
                    # Val/test: chỉ normalize, không augment
                    np.save(out_dir / fp.name, normalize_sequence(seq, T))

        n_train_aug = len(split_map['train']) * 5
        n_v = len(split_map['val'])
        n_t = len(split_map['test'])
        print(f"  ✓ {label:20s}  train={n_train_aug}  val={n_v}  test={n_t}")
        total_train += n_train_aug
        total_val   += n_v
        total_test  += n_t

    print(f"\n{'='*45}")
    print(f"  Tổng train : {total_train}")
    print(f"  Tổng val   : {total_val}")
    print(f"  Tổng test  : {total_test}")
    print(f"  Lưu tại    : {proc_path.resolve()}")

    # Ghi lại labels file theo thứ tự đã sort
    (proc_path / "vsl_labels.txt").write_text("\n".join(labels), encoding='utf-8')
    print(f"  Labels     : {proc_path / 'vsl_labels.txt'}")


if __name__ == "__main__":
    process(
        raw_dir    = "data/raw",
        proc_dir   = "data/processed",
        T          = 64,
        train_ratio= 0.70,
        val_ratio  = 0.15,
    )
