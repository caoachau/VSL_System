"""
models/ensemble.py
Inference engine: ensemble ST-GCN + CTR-GCN real-time với sliding buffer.
"""
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

from preprocessing.mediapipe_extractor import SkeletonExtractor
from preprocessing.normalizer import normalize_sequence


class EnsembleInference:
    def __init__(self,
                 stgcn,
                 ctrgcn,
                 labels:    list[str],
                 T:         int   = 64,
                 device:    str   = 'cpu',
                 smoothing: int   = 5,
                 w_stgcn:   float = 0.45,
                 w_ctrgcn:  float = 0.55):
        """
        Args:
            smoothing : số frame dùng để smooth kết quả (voting)
            w_stgcn   : trọng số ST-GCN trong ensemble
            w_ctrgcn  : trọng số CTR-GCN trong ensemble
        """
        self.stgcn   = stgcn.eval().to(device)
        self.ctrgcn  = ctrgcn.eval().to(device)
        self.labels  = labels
        self.T       = T
        self.device  = device
        self.w_stgcn = w_stgcn
        self.w_ctrgcn= w_ctrgcn

        self.extractor = SkeletonExtractor()
        self.buffer    = deque(maxlen=T)                # sliding window
        self.pred_hist = deque(maxlen=smoothing)        # temporal smoothing

    def push_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Nhận frame BGR từ webcam, extract skeleton, đẩy vào buffer.
        Trả về (keypoints, annotated_frame)
        """
        kp, annotated, _ = self.extractor.extract_with_draw(frame_bgr)
        self.buffer.append(kp)
        return kp, annotated

    def predict(self, top_k: int = 3) -> list[tuple[str, float]] | None:
        """
        Dự đoán từ sliding window hiện tại.
        Trả về [(label, score), ...] hoặc None nếu chưa đủ frame.
        """
        if len(self.buffer) < self.T // 2:
            return None

        # Build sequence từ buffer (pad nếu cần)
        seq = np.stack(self.buffer)         # (N, 75, 3)
        seq = normalize_sequence(seq, self.T)  # (T, 75, 3)

        # (T, J, C) → (1, C, T, J)
        x = torch.from_numpy(seq).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            p1 = F.softmax(self.stgcn(x),  dim=-1)
            p2 = F.softmax(self.ctrgcn(x), dim=-1)
            probs = self.w_stgcn * p1 + self.w_ctrgcn * p2  # (1, n_classes)

        probs_np = probs[0].cpu().numpy()

        # Temporal smoothing: average over recent predictions
        self.pred_hist.append(probs_np)
        smooth_probs = np.mean(self.pred_hist, axis=0)

        # Top-K
        top_indices = np.argsort(smooth_probs)[::-1][:top_k]
        return [(self.labels[i], float(smooth_probs[i])) for i in top_indices]

    def reset(self):
        """Xóa buffer khi cần restart nhận dạng"""
        self.buffer.clear()
        self.pred_hist.clear()

    def close(self):
        self.extractor.close()
