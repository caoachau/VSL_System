# reset.py — chạy: python reset.py
import shutil
from pathlib import Path

def reset(confirm=True):
    targets = {
        "weights/":           "Model weights (STGCN, CTR-GCN)",
        "data/raw/":          "Dữ liệu thô (video keypoints)",
        "data/processed/":    "Dữ liệu đã augment + split",
        "outputs/":           "Lịch sử export CSV, confusion matrix",
    }

    print("=== VSL System Reset ===\n")
    print("Chọn thứ muốn reset:")
    print("  1. Chỉ xóa weights  (giữ lại data)")
    print("  2. Chỉ xóa data     (giữ lại weights)")
    print("  3. Xóa tất cả       (về trạng thái ban đầu hoàn toàn)")
    print("  0. Hủy\n")

    choice = input("Nhập lựa chọn (0/1/2/3): ").strip()

    if choice == '0':
        print("Đã hủy."); return

    to_delete = []
    if choice == '1':
        to_delete = ["weights/"]
    elif choice == '2':
        to_delete = ["data/raw/", "data/processed/"]
    elif choice == '3':
        to_delete = list(targets.keys())
    else:
        print("Lựa chọn không hợp lệ."); return

    print("\nSẽ xóa:")
    for d in to_delete:
        print(f"  ✗ {d}  ({targets[d]})")

    confirm = input("\nXác nhận? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Đã hủy."); return

    for d in to_delete:
        p = Path(d)
        if p.exists():
            shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)  # Tạo lại thư mục rỗng
            print(f"  ✓ Đã xóa và tạo lại: {d}")
        else:
            print(f"  - Không tồn tại: {d}")

    print("\n✓ Reset hoàn tất!")
    print("  Bước tiếp theo: python data_collection/collector.py")

if __name__ == "__main__":
    reset()