"""Claude API → GPT-SoVITS (huzisaki voice) 音声チャット。

挙動:
  - 標準入力に質問を入れる
  - Claude の応答を全部受け取ってから TTS に投げる(非ストリーミング)
  - 生成された音声を再生
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
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


class LLMClient:
    def chat(self, history: list[dict]) -> str:
        raise NotImplementedError


class AnthropicClient(LLMClient):
    def __init__(self) -> None:
        from anthropic import Anthropic

        self.client = Anthropic()
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    def chat(self, history: list[dict]) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=history,
        )
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAIClient(LLMClient):
    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def chat(self, history: list[dict]) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=512,
            messages=messages,
        )
        return resp.choices[0].message.content or ""


def build_llm() -> LLMClient:
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY を設定してください", file=sys.stderr)
            sys.exit(1)
        return AnthropicClient()
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY を設定してください", file=sys.stderr)
            sys.exit(1)
        return OpenAIClient()
    print(f"不明な LLM_PROVIDER: {provider} (anthropic | openai)", file=sys.stderr)
    sys.exit(1)


def load_tts() -> TTS:
    cfg_path = ROOT / "src" / "tts_config.yaml"
    config = TTS_Config(str(cfg_path))
    return TTS(config)


def synthesize(tts: TTS, text: str) -> tuple[int, np.ndarray]:
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
        "streaming_mode": False,
        "parallel_infer": True,
    })
    chunks: list[np.ndarray] = []
    sr = 32000
    for s, audio in gen:
        chunks.append(audio)
        sr = s
    if not chunks:
        return sr, np.zeros(0, dtype=np.float32)
    return sr, np.concatenate(chunks)


def play(sr: int, audio: np.ndarray) -> None:
    if audio.size == 0:
        return
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    sd.play(audio, sr, blocking=True)


def main() -> None:
    llm = build_llm()
    print(f"[init] LLM: {type(llm).__name__} (model={llm.model})")

    print("[init] TTS を読み込み中…(初回は時間がかかります)")
    tts = load_tts()
    print("[init] TTS 準備完了")

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

        text = llm.chat(history)
        print(text)
        history.append({"role": "assistant", "content": text})

        if text.strip():
            sr, audio = synthesize(tts, text)
            play(sr, audio)


if __name__ == "__main__":
    main()
