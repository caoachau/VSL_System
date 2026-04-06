"""
main.py — GUI hoàn chỉnh hệ thống nhận dạng VSL
Chạy: python main.py
"""
import sys, os, threading, csv
sys.path.append(os.path.dirname(__file__))

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from datetime import datetime
from pathlib import Path
from collections import deque

from preprocessing.graph_builder import build_adjacency
from models.stgcn   import STGCN
from models.ctrgcn  import CTRGCN
from models.ensemble import EnsembleInference
from tts.speaker    import TTSSpeaker


# ── Constants ──────────────────────────────────────────────────────────
LABELS_FILE = "data/vsl_labels.txt"
WEIGHTS_DIR = "weights"
T           = 64
CAM_INDEX   = 0
CAM_W, CAM_H = 640, 480
DISPLAY_W, DISPLAY_H = 520, 390

COLORS = {
    'bg':      '#1e1e2e',
    'panel':   '#2a2a3e',
    'border':  '#3a3a5e',
    'accent':  '#7c6af7',
    'accent2': '#5eead4',
    'text':    '#cdd6f4',
    'subtext': '#a6adc8',
    'green':   '#a6e3a1',
    'yellow':  '#f9e2af',
    'red':     '#f38ba8',
    'black':   '#11111b',
}
FONT_TITLE = ('Segoe UI', 12, 'bold')
FONT_BODY  = ('Segoe UI', 10)
FONT_MONO  = ('Consolas', 10)
FONT_BIG   = ('Segoe UI', 40, 'bold')


# ── App ────────────────────────────────────────────────────────────────
class VSLApp:
    def __init__(self, root: tk.Tk, engine: EnsembleInference, tts: TTSSpeaker):
        self.root    = root
        self.engine  = engine
        self.tts     = tts

        self.running     = True
        self.paused      = False
        self.last_label  = ''
        self.history: list[tuple] = []

        # Tkinter vars
        self.auto_tts    = tk.BooleanVar(value=True)
        self.show_skel   = tk.BooleanVar(value=True)
        self.conf_thresh = tk.DoubleVar(value=0.55)

        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self._build_ui()
        self._loop()

    # ── Build UI ───────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.title("VSL Recognition System  —  RTX 4050")
        self.root.configure(bg=COLORS['bg'])
        self.root.geometry('1160x700')
        self.root.resizable(False, False)

        # ── Header ──
        header = tk.Frame(self.root, bg=COLORS['black'], height=48)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="VSL Recognition System",
                 bg=COLORS['black'], fg=COLORS['accent'],
                 font=('Segoe UI', 14, 'bold')).pack(side=tk.LEFT, padx=16, pady=10)

        self.status_dot = tk.Label(header, text="●  Đang chạy",
                                   bg=COLORS['black'], fg=COLORS['green'],
                                   font=FONT_BODY)
        self.status_dot.pack(side=tk.RIGHT, padx=16)

        # ── Main container ──
        main = tk.Frame(self.root, bg=COLORS['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # Left column: camera
        left = tk.Frame(main, bg=COLORS['bg'])
        left.pack(side=tk.LEFT, fill=tk.Y)

        self._section(left, "Camera")
        cam_frame = tk.Frame(left, bg=COLORS['black'],
                             width=DISPLAY_W, height=DISPLAY_H)
        cam_frame.pack()
        cam_frame.pack_propagate(False)

        self.video_lbl = tk.Label(cam_frame, bg=COLORS['black'])
        self.video_lbl.pack(fill=tk.BOTH, expand=True)

        # Controls row
        ctrl = tk.Frame(left, bg=COLORS['bg'])
        ctrl.pack(pady=6, fill=tk.X)
        self._btn(ctrl, "⏸  Pause",   self._toggle_pause,  COLORS['panel']).pack(side=tk.LEFT, padx=3)
        self._btn(ctrl, "🔁  Reset",   self._reset_buffer,  COLORS['panel']).pack(side=tk.LEFT, padx=3)
        self._btn(ctrl, "🔊  Đọc",    self._speak_now,     COLORS['accent']).pack(side=tk.LEFT, padx=3)
        self._btn(ctrl, "🗑  Xóa",    self._clear_history, COLORS['panel']).pack(side=tk.LEFT, padx=3)
        self._btn(ctrl, "💾  Export", self._export_csv,    COLORS['panel']).pack(side=tk.LEFT, padx=3)

        # Settings
        sett = tk.LabelFrame(left, text=" Cài đặt ",
                             bg=COLORS['bg'], fg=COLORS['subtext'],
                             font=FONT_BODY,
                             bd=1, relief=tk.FLAT)
        sett.pack(fill=tk.X, pady=4)

        row1 = tk.Frame(sett, bg=COLORS['bg']); row1.pack(fill=tk.X, padx=8, pady=3)
        tk.Checkbutton(row1, text="Tự động TTS", variable=self.auto_tts,
                       bg=COLORS['bg'], fg=COLORS['text'],
                       selectcolor=COLORS['panel'],
                       activebackground=COLORS['bg'],
                       font=FONT_BODY).pack(side=tk.LEFT)
        tk.Checkbutton(row1, text="Hiện skeleton", variable=self.show_skel,
                       bg=COLORS['bg'], fg=COLORS['text'],
                       selectcolor=COLORS['panel'],
                       activebackground=COLORS['bg'],
                       font=FONT_BODY).pack(side=tk.LEFT, padx=12)

        row2 = tk.Frame(sett, bg=COLORS['bg']); row2.pack(fill=tk.X, padx=8, pady=(0,6))
        tk.Label(row2, text="Ngưỡng:",
                 bg=COLORS['bg'], fg=COLORS['subtext'],
                 font=FONT_BODY).pack(side=tk.LEFT)
        self.thresh_lbl = tk.Label(row2, text=f"{self.conf_thresh.get():.0%}",
                                   bg=COLORS['bg'], fg=COLORS['accent'],
                                   font=FONT_BODY, width=5)
        self.thresh_lbl.pack(side=tk.RIGHT)
        tk.Scale(row2, from_=0.2, to=0.99, resolution=0.01,
                 orient=tk.HORIZONTAL, variable=self.conf_thresh,
                 bg=COLORS['bg'], fg=COLORS['text'],
                 troughcolor=COLORS['panel'], highlightthickness=0,
                 length=200, showvalue=False,
                 command=lambda v: self.thresh_lbl.config(text=f"{float(v):.0%}")
                 ).pack(side=tk.LEFT, padx=6)

        # ── Right column ──
        right = tk.Frame(main, bg=COLORS['bg'])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        # Main result card
        self._section(right, "Kết quả nhận dạng")
        card = tk.Frame(right, bg=COLORS['panel'],
                        highlightbackground=COLORS['border'],
                        highlightthickness=1)
        card.pack(fill=tk.X, pady=(0, 8))

        self.lbl_main = tk.Label(card, text="—",
                                  bg=COLORS['panel'], fg=COLORS['accent'],
                                  font=FONT_BIG)
        self.lbl_main.pack(pady=(18, 4))

        self.lbl_conf = tk.Label(card, text="Tin cậy: —",
                                  bg=COLORS['panel'], fg=COLORS['subtext'],
                                  font=FONT_BODY)
        self.lbl_conf.pack(pady=(0, 14))

        # Top-3 predictions
        self._section(right, "Top 3 dự đoán")
        top3_frame = tk.Frame(right, bg=COLORS['bg'])
        top3_frame.pack(fill=tk.X, pady=(0, 8))

        self.top3_widgets = []
        for i in range(3):
            row = tk.Frame(top3_frame, bg=COLORS['bg'])
            row.pack(fill=tk.X, pady=3)

            rank_lbl = tk.Label(row, text=f"#{i+1}",
                                bg=COLORS['bg'], fg=COLORS['subtext'],
                                font=FONT_MONO, width=3)
            rank_lbl.pack(side=tk.LEFT)

            name_lbl = tk.Label(row, text="",
                                bg=COLORS['bg'], fg=COLORS['text'],
                                font=FONT_BODY, width=18, anchor='w')
            name_lbl.pack(side=tk.LEFT)

            bar_bg = tk.Frame(row, bg=COLORS['panel'], height=14, width=220)
            bar_bg.pack(side=tk.LEFT, padx=6)
            bar_bg.pack_propagate(False)
            bar_fill = tk.Frame(bar_bg, bg=COLORS['accent'] if i == 0 else COLORS['border'],
                                height=14, width=0)
            bar_fill.place(x=0, y=0, relheight=1)

            score_lbl = tk.Label(row, text="",
                                 bg=COLORS['bg'], fg=COLORS['subtext'],
                                 font=FONT_MONO, width=6)
            score_lbl.pack(side=tk.LEFT)

            self.top3_widgets.append((name_lbl, bar_fill, bar_bg, score_lbl))

        # History
        self._section(right, "Lịch sử nhận dạng")
        self.history_box = scrolledtext.ScrolledText(
            right, height=11,
            bg=COLORS['panel'], fg=COLORS['text'],
            font=FONT_MONO, relief=tk.FLAT,
            state=tk.DISABLED,
            insertbackground=COLORS['text']
        )
        self.history_box.pack(fill=tk.BOTH, expand=True)

        # Tags
        self.history_box.tag_config('ts',    foreground=COLORS['subtext'])
        self.history_box.tag_config('label', foreground=COLORS['accent2'],
                                    font=('Consolas', 10, 'bold'))
        self.history_box.tag_config('score', foreground=COLORS['yellow'])

    def _section(self, parent, title: str):
        f = tk.Frame(parent, bg=COLORS['bg'])
        f.pack(fill=tk.X, pady=(4, 2))
        tk.Label(f, text=title,
                 bg=COLORS['bg'], fg=COLORS['subtext'],
                 font=FONT_BODY).pack(side=tk.LEFT)
        tk.Frame(f, bg=COLORS['border'], height=1).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0), pady=6)

    def _btn(self, parent, text, cmd, bg):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg=COLORS['text'],
                         activebackground=COLORS['accent'],
                         activeforeground='white',
                         font=('Segoe UI', 9), relief=tk.FLAT,
                         padx=10, pady=5, cursor='hand2', bd=0)

    # ── Main loop ──────────────────────────────────────────────────────
    def _loop(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            if not self.paused:
                kp, annotated = self.engine.push_frame(frame)
                preds = self.engine.predict(top_k=3)
                display = annotated if self.show_skel.get() else frame
                if preds:
                    self._update_results(preds)
            else:
                display = frame

            # Resize + show
            img = Image.fromarray(
                cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            ).resize((DISPLAY_W, DISPLAY_H), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(img)
            self.video_lbl.imgtk = imgtk
            self.video_lbl.config(image=imgtk)

        self.root.after(33, self._loop)  # ~30 fps

    # ── Update UI ──────────────────────────────────────────────────────
    def _update_results(self, preds: list[tuple[str, float]]):
        label, score = preds[0]
        if score < self.conf_thresh.get():
            return

        # Main card
        display_label = label.replace('_', ' ')
        self.lbl_main.config(text=display_label)
        self.lbl_conf.config(text=f"Tin cậy: {score:.1%}")

        # Top-3 bars
        bar_colors = [COLORS['accent'], COLORS['accent2'], COLORS['subtext']]
        for i, (name_w, bar_w, bg_w, score_w) in enumerate(self.top3_widgets):
            if i < len(preds):
                l, s = preds[i]
                name_w.config(text=l.replace('_',' '))
                bg_w.update_idletasks()
                w = int(bg_w.winfo_width() * s)
                bar_w.place(x=0, y=0, relheight=1, width=max(2, w))
                bar_w.config(bg=bar_colors[i])
                score_w.config(text=f"{s:.0%}")
            else:
                name_w.config(text='')
                score_w.config(text='')

        # Auto TTS (chỉ đọc khi nhãn thay đổi)
        if self.auto_tts.get() and label != self.last_label:
            self.last_label = label
            self.tts.say(display_label)
            self._add_history(display_label, score)

    def _add_history(self, label: str, score: float):
        ts = datetime.now().strftime('%H:%M:%S')
        self.history.append((ts, label, score))

        self.history_box.config(state=tk.NORMAL)
        self.history_box.insert(tk.END, f"[{ts}]  ", 'ts')
        self.history_box.insert(tk.END, f"{label:<20}", 'label')
        self.history_box.insert(tk.END, f"  {score:.1%}\n", 'score')
        self.history_box.see(tk.END)
        self.history_box.config(state=tk.DISABLED)

    # ── Controls ───────────────────────────────────────────────────────
    def _toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.status_dot.config(text="⏸  Tạm dừng", fg=COLORS['yellow'])
        else:
            self.status_dot.config(text="●  Đang chạy", fg=COLORS['green'])

    def _reset_buffer(self):
        self.engine.reset()
        self.last_label = ''
        self.lbl_main.config(text='—')
        self.lbl_conf.config(text='Tin cậy: —')

    def _speak_now(self):
        if self.last_label:
            self.tts.say_now(self.last_label)

    def _clear_history(self):
        self.history.clear()
        self.history_box.config(state=tk.NORMAL)
        self.history_box.delete('1.0', tk.END)
        self.history_box.config(state=tk.DISABLED)

    def _export_csv(self):
        if not self.history:
            messagebox.showinfo("Export", "Chưa có dữ liệu lịch sử!")
            return
        Path("outputs").mkdir(exist_ok=True)
        fname = f"outputs/history_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(fname, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(
                [('time', 'label', 'score')] + self.history)
        self.status_dot.config(text=f"✓ Exported {fname}", fg=COLORS['green'])
        messagebox.showinfo("Export", f"Đã lưu:\n{fname}")

    def destroy(self):
        self.running = False
        self.cap.release()
        self.engine.close()
        self.tts.close()
        self.root.destroy()


# ── Entry point ────────────────────────────────────────────────────────
def load_model(ModelClass, fname, A, n_classes, device):
    m = ModelClass(A, n_classes)
    path = Path(WEIGHTS_DIR) / fname
    if path.exists():
        ckpt = torch.load(path, map_location=device)
        m.load_state_dict(ckpt['model_state'])
        print(f"✓ Loaded {fname}")
    else:
        print(f"! Không tìm thấy {path} — dùng weights ngẫu nhiên (chưa train)")
    return m


def main():
    if not Path(LABELS_FILE).exists():
        print(f"✗ Không tìm thấy {LABELS_FILE}")
        sys.exit(1)

    labels    = open(LABELS_FILE, encoding='utf-8').read().splitlines()
    n_classes = len(labels)
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    A         = build_adjacency()

    print(f"Device   : {device}")
    print(f"Labels   : {n_classes} từ")

    stgcn  = load_model(STGCN,   "STGCN_best.pth",  A, n_classes, device)
    ctrgcn = load_model(CTRGCN,  "CTRGCN_best.pth", A, n_classes, device)

    engine = EnsembleInference(stgcn, ctrgcn, labels, T=T, device=device)
    tts    = TTSSpeaker(lang='vi')

    root = tk.Tk()
    app  = VSLApp(root, engine, tts)
    root.protocol("WM_DELETE_WINDOW", app.destroy)
    root.mainloop()


if __name__ == '__main__':
    main()
