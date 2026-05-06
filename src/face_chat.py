"""アルターエゴ GUI: 全画面の顔画像 + 上部にオーバーレイ入力 + 透過チャットログ。

仕様:
  - フルスクリーン表示 (Esc で切替)
  - 顔画像は cover スケールで画面全面 (LLM 応答先頭の [NN] で切替)
  - 入力は画面上部にオーバーレイ (透過は不可: tkinter Entry の制約)
  - レスポンス (チャットログ) は背景画像と同じ Canvas に直接 create_text で
    描画するため真の透過になる
  - 送受信のみを GUI 表示。system/error/face ログは標準出力
  - 応答テキストは GPT-SoVITS で再生 (TTS 準備完了まで音声は出ない)
"""

from __future__ import annotations

import queue
import re
import sys
import threading
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from src.voice_chat import (
    ROOT,
    SYSTEM_PROMPT as PERSONA_PROMPT,
    build_llm,
    load_tts,
    play,
    synthesize,
)

FACE_DIR = ROOT / "face"
DEFAULT_FACE = 0

FACE_LABELS = {
    0: "微笑み", 1: "嬉しい", 2: "にっこり", 3: "驚き",
    4: "焦り", 5: "ひらめき", 6: "落ち込み", 7: "びっくり",
    8: "慌て", 9: "号泣", 10: "静かに涙", 11: "涙目",
    12: "不安", 13: "大笑い", 14: "考え中", 15: "すすり泣き",
}

FACE_TAG_INSTRUCTION = (
    "\n\n# 表情タグ (この実行環境で必須)\n"
    "返答の先頭に必ず [NN] (00〜15) の表情番号を付けてください。例: 「[02] こんにちは、ご主人タマ♪」\n"
    "00=微笑み, 01=嬉しい, 02=にっこり, 03=驚き, 04=焦り, 05=ひらめき, "
    "06=落ち込み, 07=びっくり, 08=慌て, 09=号泣, 10=静かに涙, 11=涙目, "
    "12=不安, 13=大笑い, 14=考え中, 15=すすり泣き"
)
GUI_SYSTEM_PROMPT = PERSONA_PROMPT + FACE_TAG_INSTRUCTION

FACE_TAG_RE = re.compile(r"^\s*[\[［](\d{1,2})[\]］]\s*")

LOG_MAX_MESSAGES = 6  # 直近 N 件だけ画面に残す
LOG_FONT = ("Helvetica", 14)
LOG_TOP = 80          # 入力欄の下から開始する y 座標
LOG_GAP = 36          # 各メッセージの間隔 (px)
LOG_REL_WIDTH = 0.7   # 画面幅に対するログ領域の比率
LOG_SHADOW = "#000"   # 文字を読みやすくする縁取り色


def parse_face(text: str) -> tuple[int, str]:
    m = FACE_TAG_RE.match(text)
    if m:
        face = int(m.group(1))
        if 0 <= face <= 15:
            return face, text[m.end():]
    return DEFAULT_FACE, text


class FaceChatApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("alter_ego")
        self.root.configure(bg="#000")
        try:
            self.root.attributes("-fullscreen", True)
        except tk.TclError:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            self.root.geometry(f"{sw}x{sh}+0+0")
        self.root.bind("<Escape>", lambda e: self._toggle_fullscreen())
        self._fullscreen = True

        # 顔画像 + チャットログ用の単一 Canvas (画面いっぱい)
        self.canvas = tk.Canvas(
            root, bg="#000", highlightthickness=0, bd=0
        )
        self.canvas.place(x=0, y=0, relwidth=1, relheight=1)
        self.face_item = self.canvas.create_image(0, 0, anchor="nw")

        # 入力欄 (透過不可なので最小限のスタイル)
        input_frame = tk.Frame(root, bg="#000", bd=0)
        input_frame.place(relx=0.5, y=20, anchor="n", relwidth=0.7)
        self.entry = tk.Entry(
            input_frame,
            font=("Helvetica", 16),
            bg="#0d0d0d",
            fg="#fff",
            insertbackground="#fff",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#444",
            highlightcolor="#888",
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 8))
        self.entry.bind("<Return>", lambda e: self.on_send())
        self.send_btn = ttk.Button(input_frame, text="送信", command=self.on_send)
        self.send_btn.pack(side="right")

        self._face_id = -1
        self._face_size: tuple[int, int] = (0, 0)
        self._face_photo: ImageTk.PhotoImage | None = None
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # チャットログは Canvas 上のテキストアイテムとして描画 → 真の透過
        self.log_messages: list[tuple[str, str]] = []
        self._log_item_ids: list[int] = []

        self.history: list[dict] = []
        self.tts = None
        self.result_q: queue.Queue = queue.Queue()

        self.llm = build_llm(system_prompt=GUI_SYSTEM_PROMPT)
        print(f"[system] LLM: {type(self.llm).__name__} ({self.llm.model})", flush=True)
        print("[system] TTS を読み込み中…(初回は時間がかかります)", flush=True)
        threading.Thread(target=self._load_tts, daemon=True).start()

        self.root.after(100, lambda: self.set_face(DEFAULT_FACE))
        self._poll_results()
        self.entry.focus_set()

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        try:
            self.root.attributes("-fullscreen", self._fullscreen)
        except tk.TclError:
            pass

    def _load_tts(self) -> None:
        try:
            tts = load_tts()
        except Exception as e:
            self.result_q.put(("error", f"TTS load: {e}"))
            return
        self.result_q.put(("tts_ready", tts))

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self._render_face()
        self._redraw_log()

    def _render_face(self) -> None:
        if self._face_id < 0:
            return
        path = FACE_DIR / f"AlterEgoFace{self._face_id:02d}.png"
        if not path.exists():
            return
        w = max(self.canvas.winfo_width(), 1)
        h = max(self.canvas.winfo_height(), 1)
        if (w, h) == self._face_size and self._face_photo is not None:
            return
        img = Image.open(path)
        # cover: 画面いっぱいに広げてはみ出しをセンタークロップ
        iw, ih = img.size
        scale = max(w / iw, h / ih)
        nw, nh = max(int(iw * scale), w), max(int(ih * scale), h)
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        left = (nw - w) // 2
        top = (nh - h) // 2
        img = img.crop((left, top, left + w, top + h))
        photo = ImageTk.PhotoImage(img)
        self.canvas.itemconfigure(self.face_item, image=photo)
        self._face_photo = photo
        self._face_size = (w, h)
        # 顔は最背面、ログテキストはその上
        self.canvas.tag_lower(self.face_item)

    def set_face(self, face_id: int) -> None:
        if face_id == self._face_id:
            return
        self._face_id = face_id
        self._face_size = (0, 0)
        self._render_face()

    def _redraw_log(self) -> None:
        for tid in self._log_item_ids:
            self.canvas.delete(tid)
        self._log_item_ids = []

        cw = max(self.canvas.winfo_width(), 1)
        cx = cw / 2
        log_width = int(cw * LOG_REL_WIDTH)
        y = LOG_TOP

        for role, text in self.log_messages[-LOG_MAX_MESSAGES:]:
            prefix = "ご主人タマ: " if role == "user" else "アルターエゴ: "
            full = prefix + text
            color = "#ffe27a" if role == "user" else "#ffffff"
            # 画像の上で読めるように 1px の影を縁取り風に置く
            shadow_offsets = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
            for dx, dy in shadow_offsets:
                sid = self.canvas.create_text(
                    cx + dx, y + dy,
                    anchor="n",
                    text=full,
                    fill=LOG_SHADOW,
                    font=LOG_FONT,
                    width=log_width,
                )
                self._log_item_ids.append(sid)
            tid = self.canvas.create_text(
                cx, y,
                anchor="n",
                text=full,
                fill=color,
                font=LOG_FONT,
                width=log_width,
            )
            self._log_item_ids.append(tid)
            bbox = self.canvas.bbox(tid)
            if bbox:
                y = bbox[3] + LOG_GAP
            else:
                y += LOG_GAP

    def _append_log(self, role: str, text: str) -> None:
        if role not in ("user", "assistant"):
            print(f"[{role}] {text}", flush=True)
            return
        self.log_messages.append((role, text))
        if len(self.log_messages) > LOG_MAX_MESSAGES:
            self.log_messages = self.log_messages[-LOG_MAX_MESSAGES:]
        self._redraw_log()

    def on_send(self) -> None:
        msg = self.entry.get().strip()
        if not msg:
            return
        self.entry.delete(0, "end")
        self._append_log("user", msg)
        self.history.append({"role": "user", "content": msg})
        self.send_btn.configure(state="disabled")
        threading.Thread(
            target=self._reply, args=(list(self.history),), daemon=True
        ).start()

    def _reply(self, history: list[dict]) -> None:
        try:
            text = self.llm.chat(history)
        except Exception as e:
            self.result_q.put(("error", str(e)))
            return
        self.result_q.put(("reply", text))

    def _poll_results(self) -> None:
        try:
            while True:
                kind, payload = self.result_q.get_nowait()
                self._handle_result(kind, payload)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_results)

    def _handle_result(self, kind: str, payload) -> None:
        if kind == "tts_ready":
            self.tts = payload
            print("[system] TTS 準備完了", flush=True)
        elif kind == "reply":
            face, body = parse_face(payload)
            label = FACE_LABELS.get(face, "?")
            self.set_face(face)
            print(f"[face] {face:02d} {label}", flush=True)
            self._append_log("assistant", body)
            self.history.append({"role": "assistant", "content": payload})
            self.send_btn.configure(state="normal")
            self.entry.focus_set()
            if self.tts and body.strip():
                threading.Thread(target=self._speak, args=(body,), daemon=True).start()
        elif kind == "error":
            print(f"[error] {payload}", flush=True, file=sys.stderr)
            self.send_btn.configure(state="normal")

    def _speak(self, text: str) -> None:
        try:
            sr, audio = synthesize(self.tts, text)
            play(sr, audio)
        except Exception as e:
            self.result_q.put(("error", f"TTS: {e}"))


def main() -> None:
    root = tk.Tk()
    FaceChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
