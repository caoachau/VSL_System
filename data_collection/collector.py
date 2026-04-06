"""
data_collection/collector.py
Thu thập dữ liệu VSL qua webcam.

Điều khiển:
  SPACE  — Bắt đầu quay 1 clip (64 frame ~ 2 giây)
  Q      — Bỏ qua, sang từ tiếp theo
  R      — Xem lại số clip đã quay từ hiện tại
  ESC    — Thoát và lưu tiến trình

Chạy: python data_collection/collector.py
"""
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.mediapipe_extractor import SkeletonExtractor


class VSLCollector:
    def __init__(self,
                 labels_file:      str = "data/vsl_labels.txt",
                 output_dir:       str = "data/raw",
                 clips_per_label:  int = 30,
                 T:                int = 64,
                 fps_target:       int = 30,
                 camera_index:     int = 0):

        self.labels          = open(labels_file, encoding='utf-8').read().splitlines()
        self.output_dir      = Path(output_dir)
        self.clips_per_label = clips_per_label
        self.T               = T
        self.fps_target      = fps_target
        self.camera_index    = camera_index
        self.extractor       = SkeletonExtractor()

        # Màu sắc giao diện
        self.C = {
            'bg':      (30,  30,  46),
            'green':   (100, 221, 150),
            'red':     (240, 100, 100),
            'yellow':  (240, 200,  80),
            'white':   (220, 220, 230),
            'gray':    (140, 140, 160),
            'accent':  (140, 120, 250),
        }

    # ── Helpers ──────────────────────────────────────────────────────
    def _text(self, frame, text, pos, color, scale=0.7, thickness=1):
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)

    def _overlay(self, frame, alpha=0.6):
        """Vẽ thanh header mờ phía trên"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (frame.shape[1], 90), (20,20,35), -1)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

    def _progress_bar(self, frame, current, total, y=95, height=6):
        W = frame.shape[1]
        cv2.rectangle(frame, (0, y), (W, y+height), (50,50,70), -1)
        fill = int(W * current / total) if total > 0 else 0
        cv2.rectangle(frame, (0, y), (fill, y+height), self.C['accent'], -1)

    def _detect_hands(self, frame):
        """Trả về True nếu detect được ít nhất 1 tay"""
        kp = self.extractor.extract(frame)
        left_ok  = kp[33:54].any()
        right_ok = kp[54:75].any()
        return left_ok or right_ok

    # ── Record 1 clip ────────────────────────────────────────────────
    def _record_clip(self, cap, label: str) -> np.ndarray | None:
        """Quay đúng T frames, hiển thị countdown, trả về (T, 75, 3)"""
        frames_kp = []
        frame_delay = max(1, int(1000 / self.fps_target))

        for i in range(self.T):
            ret, frame = cap.read()
            if not ret:
                return None

            frame = cv2.flip(frame, 1)
            kp, annotated, _ = self.extractor.extract_with_draw(frame)
            frames_kp.append(kp)

            # UI trong lúc quay
            self._overlay(annotated)
            self._text(annotated, f"TU: {label}", (12, 35),
                       self.C['accent'], scale=1.0, thickness=2)
            self._text(annotated, f"DANG QUAY  {i+1}/{self.T}",
                       (12, 68), self.C['red'], scale=0.65)

            # Vòng tròn đếm
            progress = int(360 * i / self.T)
            cv2.ellipse(annotated, (frame.shape[1]-50, 50),
                        (30, 30), -90, 0, progress, self.C['green'], 3)
            self._text(annotated, str(self.T - i),
                       (frame.shape[1]-62, 58), self.C['white'], scale=0.7)

            cv2.imshow("VSL Collector", annotated)
            cv2.waitKey(frame_delay)

        return np.stack(frames_kp)  # (T, 75, 3)

    # ── Main collect loop ────────────────────────────────────────────
    def collect(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.fps_target)

        if not cap.isOpened():
            print("✗ Không mở được webcam!")
            return

        total_labels  = len(self.labels)
        total_clips   = 0

        for label_idx, label in enumerate(self.labels):
            save_dir = self.output_dir / label
            save_dir.mkdir(parents=True, exist_ok=True)

            # Đếm clip đã quay (nếu chạy lại)
            existing = list(save_dir.glob("*.npy"))
            clip_count = len(existing)

            print(f"\n[{label_idx+1}/{total_labels}] Từ: '{label}' "
                  f"— {clip_count}/{self.clips_per_label} clip")

            while clip_count < self.clips_per_label:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                kp, annotated, _ = self.extractor.extract_with_draw(frame)
                hand_ok = kp[33:75].any()

                # Header
                self._overlay(annotated)
                self._text(annotated, f"TU: {label}",
                            (12, 35), self.C['accent'], scale=1.0, thickness=2)
                label_info = (f"[{label_idx+1}/{total_labels}]  "
                              f"Clip {clip_count}/{self.clips_per_label}")
                self._text(annotated, label_info, (12, 68), self.C['gray'])

                # Progress bar tổng
                self._progress_bar(annotated, clip_count, self.clips_per_label)

                # Trạng thái tay
                hand_status = "Tay: OK" if hand_ok else "Tay: Khong detect duoc"
                hand_color  = self.C['green'] if hand_ok else self.C['yellow']
                self._text(annotated, hand_status,
                           (12, frame.shape[0]-60), hand_color, scale=0.6)

                # Hướng dẫn
                self._text(annotated, "SPACE: Quay  |  Q: Bo qua  |  ESC: Thoat",
                           (12, frame.shape[0]-30), self.C['gray'], scale=0.55)

                cv2.imshow("VSL Collector", annotated)
                key = cv2.waitKey(1) & 0xFF

                if key == 27:   # ESC
                    print(f"\nThoát sớm. Tổng: {total_clips} clip đã lưu.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                elif key == ord('q') or key == ord('Q'):
                    print(f"  Bỏ qua từ '{label}'")
                    break

                elif key == ord(' '):
                    # Cảnh báo nếu không detect tay
                    if not hand_ok:
                        print(f"  Cảnh báo: Không thấy tay! Vẫn quay...")

                    seq = self._record_clip(cap, label)
                    if seq is not None:
                        fname = save_dir / f"{clip_count:04d}.npy"
                        np.save(fname, seq)
                        clip_count += 1
                        total_clips += 1
                        print(f"  ✓ Clip {clip_count}/{self.clips_per_label} — {fname.name}")

                        # Flash xanh báo thành công
                        ok_frame = np.zeros_like(frame)
                        ok_frame[:] = (30, 80, 30)
                        self._text(ok_frame, "DA LUU!", (200, 250),
                                   self.C['green'], scale=2.0, thickness=3)
                        cv2.imshow("VSL Collector", ok_frame)
                        cv2.waitKey(400)

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Hoàn thành! Tổng cộng {total_clips} clip mới đã lưu.")
        print(f"  Thư mục: {self.output_dir.resolve()}")


if __name__ == "__main__":
    collector = VSLCollector(
        labels_file     = "data/vsl_labels.txt",
        output_dir      = "data/raw",
        clips_per_label = 30,
        T               = 64,
    )
    collector.collect()
