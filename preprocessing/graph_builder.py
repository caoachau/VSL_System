"""
preprocessing/graph_builder.py
Xây dựng adjacency matrix cho skeleton 75 joints:
  0-32  : Pose (33 joints)
  33-53 : Left hand (21 joints)
  54-74 : Right hand (21 joints)
"""
import numpy as np

# Pose edges (MediaPipe Pose 33 joints)
POSE_EDGES = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]

# Hand edges (21 joints, dùng cho cả 2 tay)
HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),        # ngón cái
    (0,5),(5,6),(6,7),(7,8),        # ngón trỏ
    (0,9),(9,10),(10,11),(11,12),   # ngón giữa
    (0,13),(13,14),(14,15),(15,16), # ngón áp út
    (0,17),(17,18),(18,19),(19,20), # ngón út
    (5,9),(9,13),(13,17),           # ngang lòng bàn tay
]

def build_adjacency(n_joints: int = 75) -> np.ndarray:
    """
    Trả về adjacency matrix đã normalize theo D^{-1/2} A D^{-1/2}
    Shape: (75, 75), dtype float32
    """
    A = np.eye(n_joints, dtype=np.float32)

    # Pose
    for i, j in POSE_EDGES:
        A[i, j] = A[j, i] = 1.0

    # Left hand (offset 33)
    for i, j in HAND_EDGES:
        A[33+i, 33+j] = A[33+j, 33+i] = 1.0

    # Right hand (offset 54)
    for i, j in HAND_EDGES:
        A[54+i, 54+j] = A[54+j, 54+i] = 1.0

    # Kết nối tay với cổ tay pose
    A[15, 33] = A[33, 15] = 1.0  # left wrist  ↔ left hand root
    A[16, 54] = A[54, 16] = 1.0  # right wrist ↔ right hand root

    # Symmetric normalized: D^{-1/2} A D^{-1/2}
    D_inv_sqrt = np.diag(A.sum(axis=1) ** -0.5)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm.astype(np.float32)


if __name__ == "__main__":
    A = build_adjacency()
    print(f"Adjacency matrix shape: {A.shape}")
    print(f"Non-zero entries     : {(A > 0).sum()}")
    print(f"Value range          : [{A.min():.4f}, {A.max():.4f}]")
