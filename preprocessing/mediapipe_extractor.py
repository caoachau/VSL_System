"""
preprocessing/mediapipe_extractor.py
Trích xuất skeleton keypoints từ frame BGR dùng MediaPipe Holistic.
Output: np.ndarray shape (75, 3) — [x, y, z] normalized [0,1]
"""
import cv2
import mediapipe as mp
import numpy as np


class SkeletonExtractor:
    N_POSE  = 33
    N_HAND  = 21
    N_TOTAL = 75  # 33 + 21 + 21

    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence:  float = 0.5):
        self.holistic = mp.solutions.holistic.Holistic(
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Args:
            frame_bgr: frame từ OpenCV (H, W, 3) BGR
        Returns:
            keypoints: (75, 3) float32 — zeros nếu không detect được
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.holistic.process(rgb)
        rgb.flags.writeable = True

        kp = np.zeros((self.N_TOTAL, 3), dtype=np.float32)

        # Pose: joints 0–32
        if result.pose_landmarks:
            for i, lm in enumerate(result.pose_landmarks.landmark[:self.N_POSE]):
                kp[i] = [lm.x, lm.y, lm.z]

        # Left hand: joints 33–53
        if result.left_hand_landmarks:
            for i, lm in enumerate(result.left_hand_landmarks.landmark):
                kp[33 + i] = [lm.x, lm.y, lm.z]

        # Right hand: joints 54–74
        if result.right_hand_landmarks:
            for i, lm in enumerate(result.right_hand_landmarks.landmark):
                kp[54 + i] = [lm.x, lm.y, lm.z]

        return kp

    def draw_landmarks(self, frame_bgr: np.ndarray, result) -> np.ndarray:
        """Vẽ skeleton lên frame để hiển thị (dùng trong GUI)"""
        annotated = frame_bgr.copy()
        if result.pose_landmarks:
            self.mp_draw.draw_landmarks(
                annotated, result.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw.DrawingSpec(
                    color=(80,200,120), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_draw.DrawingSpec(
                    color=(80,200,120), thickness=1)
            )
        for hand_lm in [result.left_hand_landmarks, result.right_hand_landmarks]:
            if hand_lm:
                self.mp_draw.draw_landmarks(
                    annotated, hand_lm,
                    mp.solutions.holistic.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style()
                )
        return annotated

    def extract_with_draw(self, frame_bgr: np.ndarray):
        """Trả về (keypoints, annotated_frame, raw_result)"""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.holistic.process(rgb)
        rgb.flags.writeable = True

        kp = np.zeros((self.N_TOTAL, 3), dtype=np.float32)
        if result.pose_landmarks:
            for i, lm in enumerate(result.pose_landmarks.landmark[:self.N_POSE]):
                kp[i] = [lm.x, lm.y, lm.z]
        if result.left_hand_landmarks:
            for i, lm in enumerate(result.left_hand_landmarks.landmark):
                kp[33 + i] = [lm.x, lm.y, lm.z]
        if result.right_hand_landmarks:
            for i, lm in enumerate(result.right_hand_landmarks.landmark):
                kp[54 + i] = [lm.x, lm.y, lm.z]

        annotated = self.draw_landmarks(frame_bgr, result)
        return kp, annotated, result

    def close(self):
        self.holistic.close()
