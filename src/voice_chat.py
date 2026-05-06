"""Claude API → GPT-SoVITS (huzisaki voice) ストリーミング音声チャット。

事前準備:
  - 環境変数 ANTHROPIC_API_KEY を設定
  - python -m src.voice_chat で起動

挙動:
  - 標準入力に質問を入れる
  - Claude がストリーミングでテキストを返す
  - 句点 (。?!) ごとに切って GPT-SoVITS に投げ、音声を順次再生する
"""

from __future__ import annotations

import os
import queue
import sys
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
from anthropic import Anthropic

ROOT = Path(__file__).resolve().parent.parent
# GPT-SoVITS の sv.py が os.getcwd()/GPT_SoVITS/eres2net を sys.path に足す前提なので
# モジュール import より前に GPT-SoVITS ディレクトリへ chdir する必要がある
os.chdir(ROOT / "GPT-SoVITS")
sys.path.insert(0, str(ROOT / "GPT-SoVITS"))
sys.path.insert(0, str(ROOT / "GPT-SoVITS" / "GPT_SoVITS"))

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

REF_AUDIO = ROOT / "huzisaki_models" / "huzisaki_007.wav"
REF_TEXT = "問題ないってことなんじゃないかな 犯人が証拠隠滅したってことじゃあ"
SYSTEM_PROMPT = (
    "あなたは藤崎というキャラクターとして日本語で短く自然に応答してください。"
    "1〜3文程度で簡潔に。語尾は柔らかめ。"
)
SENTENCE_DELIMS = "。?!？！\n"
MIN_CHUNK_CHARS = 8


def load_tts() -> TTS:
    cfg_path = ROOT / "src" / "tts_config.yaml"
    config = TTS_Config(str(cfg_path))
    return TTS(config)


def player_loop(audio_q: queue.Queue) -> None:
    while True:
        item = audio_q.get()
        if item is None:
            return
        sr, audio = item
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        sd.play(audio, sr, blocking=True)


def speak(tts: TTS, text: str, audio_q: queue.Queue) -> None:
    text = text.strip()
    if not text:
        return
    gen = tts.run({
        "text": text,
        "text_lang": "ja",
        "ref_audio_path": str(REF_AUDIO),
        "prompt_text": REF_TEXT,
        "prompt_lang": "ja",
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut5",
        "streaming_mode": True,
        "parallel_infer": False,
    })
    for sr, audio in gen:
        audio_q.put((sr, audio))


def chunk_at_delim(buffer: str) -> tuple[str | None, str]:
    if len(buffer) < MIN_CHUNK_CHARS:
        return None, buffer
    idx = -1
    for d in SENTENCE_DELIMS:
        i = buffer.find(d)
        if i >= 0 and (idx < 0 or i < idx):
            idx = i
    if idx < 0:
        return None, buffer
    return buffer[: idx + 1], buffer[idx + 1 :]


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY を設定してください", file=sys.stderr)
        sys.exit(1)

    print("[init] TTS を読み込み中…(初回は時間がかかります)")
    tts = load_tts()
    print("[init] TTS 準備完了")

    audio_q: queue.Queue = queue.Queue()
    threading.Thread(target=player_loop, args=(audio_q,), daemon=True).start()

    client = Anthropic()
    history: list[dict] = []

    print("\n質問を入力(空行で終了):")
    while True:
        try:
            user_msg = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_msg:
            break
        history.append({"role": "user", "content": user_msg})

        accumulated = ""
        buffer = ""
        with client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=history,
        ) as stream:
            for delta in stream.text_stream:
                print(delta, end="", flush=True)
                buffer += delta
                accumulated += delta
                while True:
                    chunk, buffer = chunk_at_delim(buffer)
                    if chunk is None:
                        break
                    speak(tts, chunk, audio_q)
        if buffer.strip():
            speak(tts, buffer, audio_q)
        print()
        history.append({"role": "assistant", "content": accumulated})

    audio_q.put(None)


if __name__ == "__main__":
    main()
