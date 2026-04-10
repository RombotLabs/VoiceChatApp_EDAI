"""
Voice Chat App
--------------
Requirements:
    pip install sounddevice soundfile numpy

Run:
    python voice_chat.py
"""
import logging
import tkinter as tk
import threading
import queue
import math
import json
import os
from src.aiutils.OllamaUtils import OllamaUtils
from src.aiutils.dir_fetcher import *
from src.aiutils.VoiceUtils import *

try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# ── Config ───────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 44100
CHANNELS     = 1
OUTPUT_FILE  = "recorded_speech.wav"
WATCH_FOLDER = "."
pwd          = os.getcwd()
data_path    = os.path.join(pwd, "data")

voice        = VoiceUtils("tts_models/en/ljspeech/tacotron2-DDC")
fetcher      = DirectoryFilesFetcher(data_path)
files        = fetcher.fetch_files()
keyword_list = "data/elite_keyword.json"
ollama       = OllamaUtils("ministral-3:3b")

logging.basicConfig(
    filename="voice_chat.log",
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s %(message)s",
)

# ── Colours ───────────────────────────────────────────────────────────────────
BG          = "#0a0a0f"
SURFACE     = "#13131a"
CARD        = "#1c1c28"
ACCENT      = "#6c63ff"
MIC_IDLE    = "#6c63ff"
MIC_RECORD  = "#ff4757"
MIC_PLAY    = "#2ed573"
MIC_PROCESS = "#ffc107"      # yellow: thinking / processing
TEXT        = "#e8e8f0"
SUBTEXT     = "#6b6b85"
BORDER      = "#2a2a3e"


# ── Lookup helper ─────────────────────────────────────────────────────────────
def look_up(expression, files):
    expression = expression.strip().lower()
    logging.info(f"Searching for: {expression}")
    results, found = [], 0
    for file in files:
        if found >= 3:
            break
        filename = os.path.basename(file).lower()
        if expression in filename:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
                found += 1
            except Exception as e:
                logging.error(f"Failed to read {file}: {e}")
    return results


# ── App ───────────────────────────────────────────────────────────────────────
class VoiceChatApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("VoiceChat")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self._recording   = False
        self._playing     = False
        self._processing  = False          # new: yellow "thinking" state
        self._frames: list = []
        self._stream      = None
        self._anim_id     = None
        self._pulse_angle = 0.0
        self._messages: list[dict] = []
        self._play_queue: queue.Queue = queue.Queue()

        self._build_ui()
        self._start_play_worker()

        self._add_message("Hold the mic button and speak. Release to stop.", "system")
        if not AUDIO_AVAILABLE:
            self._add_message(
                "⚠  sounddevice / soundfile not found.\n"
                "Run:  pip install sounddevice soundfile numpy", "system"
            )

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        W = 420
        header = tk.Frame(self.root, bg=SURFACE, height=64)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(header, text="● VOICE", bg=SURFACE, fg=ACCENT,
                 font=("Courier New", 11, "bold"), anchor="w", padx=20
                 ).pack(side="left", fill="y")

        self._status_dot = tk.Label(
            header, text="◉  idle", bg=SURFACE, fg=SUBTEXT,
            font=("Courier New", 10), anchor="e", padx=20
        )
        self._status_dot.pack(side="right", fill="y")

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        msg_frame = tk.Frame(self.root, bg=BG)
        msg_frame.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(msg_frame, bg=BG, bd=0, highlightthickness=0,
                                 width=W, height=340)
        scrollbar = tk.Scrollbar(msg_frame, orient="vertical",
                                 command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self._msg_inner = tk.Frame(self._canvas, bg=BG)
        self._canvas.create_window((0, 0), window=self._msg_inner,
                                   anchor="nw", width=W - 14)
        self._msg_inner.bind("<Configure>", self._on_frame_configure)

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        bottom = tk.Frame(self.root, bg=SURFACE, height=160)
        bottom.pack(fill="x")
        bottom.pack_propagate(False)

        self._wave_canvas = tk.Canvas(bottom, bg=SURFACE, bd=0,
                                      highlightthickness=0, width=W, height=50)
        self._wave_canvas.pack(pady=(14, 0))

        btn_frame = tk.Frame(bottom, bg=SURFACE)
        btn_frame.pack()

        self._mic_canvas = tk.Canvas(btn_frame, width=72, height=72,
                                     bg=SURFACE, bd=0, highlightthickness=0)
        self._mic_canvas.pack()
        self._draw_mic_button(MIC_IDLE, glow=False)
        self._mic_canvas.bind("<ButtonPress-1>",   self._on_mic_press)
        self._mic_canvas.bind("<ButtonRelease-1>", self._on_mic_release)

        self._hint = tk.Label(bottom, text="hold to record", bg=SURFACE,
                              fg=SUBTEXT, font=("Courier New", 9))
        self._hint.pack(pady=(6, 0))

    def _on_frame_configure(self, _event=None):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    # ── Messages ──────────────────────────────────────────────────────────────
    def _add_message(self, text: str, kind: str = "user"):
        self._messages.append({"text": text, "type": kind})
        row = tk.Frame(self._msg_inner, bg=BG, pady=6)
        row.pack(fill="x", padx=16)

        if kind == "user":
            color, fg, align, badge, bdg_col = CARD, TEXT, "e", "YOU", ACCENT
        else:
            color, fg, align, badge, bdg_col = "#0f0f1a", TEXT, "w", "SYS", SUBTEXT

        tk.Label(row, text=badge, bg=bdg_col, fg="#fff",
                 font=("Courier New", 7, "bold"), padx=5, pady=1
                 ).pack(anchor=align, pady=(0, 2))
        tk.Label(row, text=text, bg=color, fg=fg, font=("Courier New", 10),
                 wraplength=320, justify="left", padx=12, pady=10, relief="flat"
                 ).pack(anchor=align)
        self.root.after(50, lambda: self._canvas.yview_moveto(1.0))

    # ── Mic button drawing ────────────────────────────────────────────────────
    def _draw_mic_button(self, color: str, glow: bool = False):
        c = self._mic_canvas
        c.delete("all")
        cx, cy, r = 36, 36, 30
        if glow:
            for i in range(4, 0, -1):
                c.create_oval(cx-r-i*5, cy-r-i*5, cx+r+i*5, cy+r+i*5,
                              fill=self._blend(color, SURFACE, i/5), outline="")
        c.create_oval(cx-r, cy-r, cx+r, cy+r, fill=color, outline="")
        # Mic body
        c.create_rectangle(30, 18, 42, 36, fill="white", outline="", width=0)
        c.create_arc(30, 30, 42, 42, start=0, extent=-180, fill="white", outline="")
        c.create_arc(30, 12, 42, 24, start=0, extent=180,  fill="white", outline="")
        # Stand
        c.create_arc(25, 30, 47, 50, start=0, extent=-180,
                     style="arc", outline="white", width=2)
        c.create_line(36, 50, 36, 55, fill="white", width=2)
        c.create_line(30, 55, 42, 55, fill="white", width=2)

    @staticmethod
    def _blend(hex1: str, hex2: str, t: float) -> str:
        def h2r(h): return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))
        r1, g1, b1 = h2r(hex1)
        r2, g2, b2 = h2r(hex2)
        return "#{:02x}{:02x}{:02x}".format(
            int(r1+(r2-r1)*t), int(g1+(g2-g1)*t), int(b1+(b2-b1)*t))

    # ── Waveform animation ────────────────────────────────────────────────────
    def _animate(self):
        self._pulse_angle += 0.12
        c = self._wave_canvas
        W, H = 420, 50
        c.delete("all")

        if self._recording:
            colour, amp = MIC_RECORD, 14 + 8*math.sin(self._pulse_angle*1.7)
        elif self._processing:
            colour = MIC_PROCESS
            amp = 6 + 5*math.sin(self._pulse_angle*0.9)   # slow calm pulse
        elif self._playing:
            colour, amp = MIC_PLAY, 10 + 6*math.sin(self._pulse_angle*1.3)
        else:
            colour, amp = ACCENT, 4

        pts = []
        for x in range(0, W+1, 3):
            y = (H/2) + amp * math.sin(self._pulse_angle + x*0.055) * math.sin(x*0.018)
            pts.extend([x, y])
        if len(pts) >= 4:
            c.create_line(pts, fill=colour, width=2, smooth=True)

        self._anim_id = self.root.after(30, self._animate)

    def _stop_animation(self):
        if self._anim_id:
            self.root.after_cancel(self._anim_id)
            self._anim_id = None

    # ── Recording ─────────────────────────────────────────────────────────────
    def _on_mic_press(self, _event=None):
        # Block mic while processing or playing
        if not AUDIO_AVAILABLE or self._playing or self._processing:
            return
        self._recording = True
        self._frames = []
        self._draw_mic_button(MIC_RECORD, glow=True)
        self._status_dot.config(text="◉  recording …", fg=MIC_RECORD)
        self._hint.config(text="recording …")
        self._animate()

        def callback(indata, frames, time_info, status):
            if self._recording:
                self._frames.append(indata.copy())

        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                                      dtype="float32", callback=callback)
        self._stream.start()

    def _on_mic_release(self, _event=None):
        if not self._recording:
            return
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._stop_animation()
        self._wave_canvas.delete("all")

        if not self._frames:
            self._draw_mic_button(MIC_IDLE, glow=False)
            self._status_dot.config(text="◉  idle", fg=SUBTEXT)
            self._hint.config(text="hold to record")
            return

        audio = np.concatenate(self._frames, axis=0)
        sf.write(OUTPUT_FILE, audio, SAMPLE_RATE)
        self._add_message(f"🎙  Saved → {OUTPUT_FILE}", "user")

        # Kick off the heavy pipeline in a background thread so the UI stays live
        threading.Thread(target=self._process_speech, daemon=True).start()

    # ── Processing pipeline (background thread) ───────────────────────────────
    def _process_speech(self):
        # All UI changes must go through root.after() — never touch tk widgets
        # directly from a background thread.
        self.root.after(0, self._set_processing_ui, True, "processing …")

        try:
            # 1 · Transcribe ──────────────────────────────────────────────────
            self.root.after(0, self._update_hint, "transcribing …")
            question = voice.transcribe(OUTPUT_FILE)
            logging.info(f"Transcribed: {question}")
            self.root.after(0, self._add_message, f"📝  {question}", "user")

            # 2 · Extract keyword ─────────────────────────────────────────────
            self.root.after(0, self._update_hint, "extracting keyword …")
            keyword = ollama.ollama_extract_word(question, keyword_list)
            if not keyword:
                logging.error("No keyword extracted.")
                self.root.after(0, self._add_message, "⚠  No keyword found.", "system")
                return

            # 3 · Look up context ─────────────────────────────────────────────
            self.root.after(0, self._update_hint, "looking up data …")
            results = look_up(keyword, files)
            context = json.dumps(results, indent=2) if results else "No relevant data found."

            # 4 · Ask LLM ─────────────────────────────────────────────────────
            self.root.after(0, self._update_hint, "thinking …")
            prompt = (
                f"{context}\n\n"
                f"Here are wiki entries related to the question.\n"
                f"Question: {question}\n"
                f"You are a chatbot about Elite Dangerous. "
                f"Only answer the question using the provided data. "
                f"Only write simple text without any Markdown or punctuation marks."
            )
            response = ollama.ask_ollama(prompt)
            logging.info(f"Response: {response}")
            self.root.after(0, self._add_message, f"🤖  {response}", "system")

            # 5 · TTS ─────────────────────────────────────────────────────────
            self.root.after(0, self._update_hint, "generating speech …")
            voice.speak(response)

            # 6 · Queue playback — play_worker blocks mic until audio finishes
            self._play_queue.put("out.wav")

        except Exception as e:
            logging.exception("Processing error")
            self.root.after(0, self._add_message, f"⚠  Error: {e}", "system")

        finally:
            # Return to idle only if playback hasn't already taken over the UI
            self.root.after(0, self._set_processing_ui, False, "")

    # ── State helpers (always called on main thread via root.after) ───────────
    def _set_processing_ui(self, active: bool, hint_text: str):
        self._processing = active
        if active:
            self._draw_mic_button(MIC_PROCESS, glow=True)
            self._status_dot.config(text="◉  processing …", fg=MIC_PROCESS)
            self._hint.config(text=hint_text)
            self._animate()
        else:
            # Only reset if playback isn't about to take over
            if not self._playing:
                self._stop_animation()
                self._draw_mic_button(MIC_IDLE, glow=False)
                self._status_dot.config(text="◉  idle", fg=SUBTEXT)
                self._hint.config(text="hold to record")
                self._wave_canvas.delete("all")

    def _update_hint(self, text: str):
        """Update just the hint label mid-processing."""
        self._hint.config(text=text)
        self._status_dot.config(text=f"◉  {text}", fg=MIC_PROCESS)

    # ── Playback ──────────────────────────────────────────────────────────────
    def play_audio(self, filepath: str):
        """Queue an audio file. Mic is locked until playback finishes."""
        self._play_queue.put(filepath)

    def _start_play_worker(self):
        threading.Thread(target=self._play_worker, daemon=True).start()

    def _play_worker(self):
        while True:
            filepath = self._play_queue.get()
            if not os.path.exists(filepath):
                self.root.after(0, self._add_message,
                                f"⚠  File not found: {filepath}", "system")
                continue

            self._playing = True
            self.root.after(0, self._set_playing_ui, True, filepath)

            try:
                data, sr = sf.read(filepath, dtype="float32")
                sd.play(data, sr)
                sd.wait()          # blocks this worker thread only — UI stays live
            except Exception as e:
                self.root.after(0, self._add_message,
                                f"⚠  Playback error: {e}", "system")
            finally:
                self._playing = False
                self.root.after(0, self._set_playing_ui, False, filepath)

    def _set_playing_ui(self, playing: bool, filepath: str = ""):
        if playing:
            self._add_message(f"🔊  Playing → {os.path.basename(filepath)}", "system")
            self._draw_mic_button(MIC_PLAY, glow=True)
            self._status_dot.config(text="◉  playing …", fg=MIC_PLAY)
            self._hint.config(text="playing audio …")
            self._animate()
        else:
            self._stop_animation()
            self._draw_mic_button(MIC_IDLE, glow=False)
            self._status_dot.config(text="◉  idle", fg=SUBTEXT)
            self._hint.config(text="hold to record")
            self._wave_canvas.delete("all")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    root = tk.Tk()

    W, H = 420, 600
    root.geometry(f"{W}x{H}")
    root.update_idletasks()
    sx = (root.winfo_screenwidth()  - W) // 2
    sy = (root.winfo_screenheight() - H) // 2
    root.geometry(f"{W}x{H}+{sx}+{sy}")

    app = VoiceChatApp(root)

    if len(sys.argv) > 1:
        root.after(1000, lambda: app.play_audio(sys.argv[1]))

    root.mainloop()