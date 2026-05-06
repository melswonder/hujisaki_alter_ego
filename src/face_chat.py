"""藤崎の顔をメイン表示しながら下部でチャットする GUI。

仕様:
  - 上部: 表情画像 (face/AlterEgoFaceNN.png)。LLM 応答先頭の [NN] で切替
  - 中央: チャットログ (各発言を 3 行間隔で表示)
  - 下部: 入力欄 + 送信ボタン (Enter でも送信)
  - 応答テキストは GPT-SoVITS で再生 (TTS 準備完了まで音声は出ない)
"""

from __future__ import annotations

import queue
import re
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

from PIL import Image, ImageTk

from voice_chat import (
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
        self.root.geometry("760x960")

        self.face_label = tk.Label(root, bg="#111")
        self.face_label.pack(fill="both", expand=True)

        self.chat_log = scrolledtext.ScrolledText(
            root,
            height=8,
            state="disabled",
            font=("Helvetica", 13),
            wrap="word",
        )
        self.chat_log.pack(fill="x", padx=8, pady=(4, 0))

        input_frame = tk.Frame(root)
        input_frame.pack(fill="x", padx=8, pady=8)
        self.entry = tk.Entry(input_frame, font=("Helvetica", 14))
        self.entry.pack(side="left", fill="x", expand=True, ipady=4)
        self.entry.bind("<Return>", lambda e: self.on_send())
        self.send_btn = ttk.Button(input_frame, text="送信", command=self.on_send)
        self.send_btn.pack(side="right", padx=(8, 0))

        self._face_id = -1
        self._face_size: tuple[int, int] = (0, 0)
        self.root.bind("<Configure>", self._on_resize)

        self.history: list[dict] = []
        self.tts = None
        self.result_q: queue.Queue = queue.Queue()

        self.llm = build_llm(system_prompt=GUI_SYSTEM_PROMPT)
        self._append_log("system", f"LLM: {type(self.llm).__name__} ({self.llm.model})")
        self._append_log("system", "TTS を読み込み中…(初回は時間がかかります)")
        threading.Thread(target=self._load_tts, daemon=True).start()

        self.root.after(100, lambda: self.set_face(DEFAULT_FACE))
        self._poll_results()
        self.entry.focus_set()

    def _load_tts(self) -> None:
        try:
            tts = load_tts()
        except Exception as e:
            self.result_q.put(("error", f"TTS load: {e}"))
            return
        self.result_q.put(("tts_ready", tts))

    def _on_resize(self, event: tk.Event) -> None:
        if event.widget is self.root:
            self.root.after_idle(self._render_face)

    def _render_face(self) -> None:
        if self._face_id < 0:
            return
        path = FACE_DIR / f"AlterEgoFace{self._face_id:02d}.png"
        if not path.exists():
            return
        w = max(self.face_label.winfo_width(), 1)
        h = max(self.face_label.winfo_height(), 1)
        if (w, h) == self._face_size and getattr(self.face_label, "_photo", None):
            return
        img = Image.open(path)
        img.thumbnail((w, h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.face_label.configure(image=photo)
        self.face_label._photo = photo  # keep ref
        self._face_size = (w, h)

    def set_face(self, face_id: int) -> None:
        if face_id == self._face_id:
            return
        self._face_id = face_id
        self._face_size = (0, 0)
        self._render_face()

    def _append_log(self, role: str, text: str) -> None:
        self.chat_log.config(state="normal")
        if self.chat_log.index("end-1c") != "1.0":
            self.chat_log.insert("end", "\n\n\n")  # 3 行間隔
        prefix = {
            "user": "ご主人タマ: ",
            "assistant": "アルターエゴ: ",
            "system": "[system] ",
        }[role]
        self.chat_log.insert("end", prefix + text)
        self.chat_log.see("end")
        self.chat_log.config(state="disabled")

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
            self._append_log("system", "TTS 準備完了")
        elif kind == "reply":
            face, body = parse_face(payload)
            label = FACE_LABELS.get(face, "?")
            self.set_face(face)
            self._append_log("assistant", body)
            self._append_log("system", f"表情: {face:02d} {label}")
            self.history.append({"role": "assistant", "content": payload})
            self.send_btn.configure(state="normal")
            self.entry.focus_set()
            if self.tts and body.strip():
                threading.Thread(target=self._speak, args=(body,), daemon=True).start()
        elif kind == "error":
            self._append_log("system", f"エラー: {payload}")
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
