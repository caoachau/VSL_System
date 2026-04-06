"""
check_cuda.py — Kiểm tra môi trường trước khi chạy hệ thống VSL
Chạy: python check_cuda.py
"""
import sys

def check_python():
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 9
    print(f"{'✓' if ok else '✗'} Python {v.major}.{v.minor}.{v.micro}"
          + ("" if ok else "  → cần Python 3.9+"))
    return ok

def check_torch():
    try:
        import torch
        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "N/A"
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if cuda else 0
        print(f"{'✓' if cuda else '!'} PyTorch {torch.__version__} | "
              f"CUDA {'available' if cuda else 'NOT found'}")
        if cuda:
            print(f"  GPU : {name}")
            print(f"  VRAM: {vram} GB")
        else:
            print("  → Sẽ train trên CPU (chậm hơn ~10x)")
        return True
    except ImportError:
        print("✗ PyTorch chưa cài — chạy: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return False

def check_mediapipe():
    try:
        import mediapipe as mp
        print(f"✓ MediaPipe {mp.__version__}")
        return True
    except ImportError:
        print("✗ MediaPipe chưa cài — chạy: pip install mediapipe")
        return False

def check_opencv():
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ok = cap.isOpened()
        cap.release()
        print(f"{'✓' if ok else '!'} OpenCV {cv2.__version__} | Webcam: {'detected' if ok else 'not found'}")
        return True
    except ImportError:
        print("✗ OpenCV chưa cài — chạy: pip install opencv-python")
        return False

def check_others():
    libs = [('gtts','gTTS'),('pygame','pygame'),
            ('PIL','Pillow'),('sklearn','scikit-learn'),('tqdm','tqdm')]
    for mod, name in libs:
        try:
            __import__(mod)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} chưa cài")

if __name__ == "__main__":
    print("=" * 50)
    print("  VSL System — Kiểm tra môi trường")
    print("=" * 50)
    check_python()
    check_torch()
    check_mediapipe()
    check_opencv()
    check_others()
    print("=" * 50)
    print("Nếu tất cả ✓ → chạy: python data_collection/collector.py")
