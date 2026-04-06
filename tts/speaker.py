"""
tts/speaker.py — Text-to-Speech tiếng Việt dùng gTTS + pygame
Chạy trong thread riêng để không block GUI.
"""
import threading
import queue
import tempfile
import os
import pygame
from gtts import gTTS


class TTSSpeaker:
    def __init__(self, lang: str = 'vi', slow: bool = False):
        self.lang  = lang
        self.slow  = slow
        self._q    = queue.Queue()
        self._lock = threading.Lock()

        pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=512)
        pygame.mixer.init()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def say(self, text: str):
        """Đưa text vào queue để đọc — không block."""
        if text and text.strip():
            self._q.put(text.strip())

    def say_now(self, text: str):
        """Xóa queue cũ, đọc ngay text mới."""
        while not self._q.empty():
            try: self._q.get_nowait()
            except queue.Empty: break
        self.say(text)

    def _worker(self):
        while True:
            text = self._q.get()
            tmp = None
            try:
                tts = gTTS(text=text, lang=self.lang, slow=self.slow)
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                    tts.save(f.name)
                    tmp = f.name

                with self._lock:
                    pygame.mixer.music.load(tmp)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.wait(50)

            except Exception as e:
                print(f"[TTS] Error: {e}")
            finally:
                if tmp and os.path.exists(tmp):
                    try: os.unlink(tmp)
                    except: pass

    def stop(self):
        with self._lock:
            pygame.mixer.music.stop()

    def close(self):
        self.stop()
        pygame.mixer.quit()
