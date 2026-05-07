"""Claude API → GPT-SoVITS (huzisaki voice) 音声チャット。

挙動:
  - 標準入力に質問を入れる
  - LLM の応答をストリーミングで受け取り、句点 (。 ! ? ！ ？ 改行) ごとに
    1 文ずつ TTS に投入する
  - 合成は別スレッド、再生はさらに別スレッドで行うので、文 N の再生中に
    文 N+1 の合成を進められる

環境変数:
  - LLM_PROVIDER       anthropic | openai (既定: anthropic)
  - ANTHROPIC_MODEL    既定: claude-sonnet-4-5
  - OPENAI_MODEL       既定: gpt-4o-mini
  - TTS_COMPILE        1 にすると BERT/CN-HuBERT に torch.compile を試す
  - TTS_COMPILE_BACKEND torch.compile の backend (既定: inductor)
  - TTS_COMPILE_MODE    torch.compile の mode    (既定: default)
"""

from __future__ import annotations

import os
import queue
import re
import sys
import threading
from collections.abc import Iterator
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
PERSONA_PATH = ROOT / "src" / "persona.md"
SYSTEM_PROMPT = (
    PERSONA_PATH.read_text(encoding="utf-8")
    if PERSONA_PATH.exists()
    else "あなたは藤崎というキャラクターとして日本語で短く自然に応答してください。"
)


class LLMClient:
    system_prompt: str = SYSTEM_PROMPT
    model: str = ""

    def chat_stream(self, history: list[dict]) -> Iterator[str]:
        raise NotImplementedError


class AnthropicClient(LLMClient):
    def __init__(self, system_prompt: str = SYSTEM_PROMPT) -> None:
        from anthropic import Anthropic

        self.client = Anthropic()
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        self.system_prompt = system_prompt

    def chat_stream(self, history: list[dict]) -> Iterator[str]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=self.system_prompt,
            messages=history,
        ) as stream:
            for text in stream.text_stream:
                yield text


class OpenAIClient(LLMClient):
    def __init__(self, system_prompt: str = SYSTEM_PROMPT) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.system_prompt = system_prompt

    def chat_stream(self, history: list[dict]) -> Iterator[str]:
        messages = [{"role": "system", "content": self.system_prompt}, *history]
        stream = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def build_llm(system_prompt: str = SYSTEM_PROMPT) -> LLMClient:
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY を設定してください", file=sys.stderr)
            sys.exit(1)
        return AnthropicClient(system_prompt)
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY を設定してください", file=sys.stderr)
            sys.exit(1)
        return OpenAIClient(system_prompt)
    print(f"不明な LLM_PROVIDER: {provider} (anthropic | openai)", file=sys.stderr)
    sys.exit(1)


# 「。！？!?\n」の連続を1つの区切りとして扱う
_SENTENCE_END_RE = re.compile(r"[。！？!?\n]+")


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """buffer を句点で区切り (確定文のリスト, 残り) を返す。

    例: "こんにちは。元気？まだ" → (["こんにちは。", "元気？"], "まだ")
    """
    sentences: list[str] = []
    last = 0
    for m in _SENTENCE_END_RE.finditer(buffer):
        sentences.append(buffer[last : m.end()])
        last = m.end()
    return sentences, buffer[last:]


def _maybe_torch_compile(tts: TTS) -> None:
    """TTS_COMPILE=1 のとき BERT / CN-HuBERT に torch.compile をかける。

    t2s_model / vits_model は実行時に属性を差し替える箇所があり compile と相性が悪い
    ので、副作用が少ない 2 つだけを対象にする。失敗してもログを出して続行。
    """
    flag = os.environ.get("TTS_COMPILE", "").lower()
    if flag not in ("1", "true", "yes"):
        return
    try:
        import torch
    except ImportError:
        return

    backend = os.environ.get("TTS_COMPILE_BACKEND", "inductor")
    mode = os.environ.get("TTS_COMPILE_MODE", "default")

    def _compile(name: str, model):
        try:
            return torch.compile(model, backend=backend, mode=mode)
        except Exception as e:
            print(f"[tts] torch.compile failed for {name}: {e}", flush=True)
            return None

    if tts.bert_model is not None:
        compiled = _compile("bert_model", tts.bert_model)
        if compiled is not None:
            tts.bert_model = compiled
            tts.text_preprocessor.bert_model = compiled
            print(f"[tts] torch.compile applied to bert_model "
                  f"(backend={backend}, mode={mode})", flush=True)
    if tts.cnhuhbert_model is not None and hasattr(tts.cnhuhbert_model, "model"):
        compiled = _compile("cnhuhbert.model", tts.cnhuhbert_model.model)
        if compiled is not None:
            tts.cnhuhbert_model.model = compiled
            print(f"[tts] torch.compile applied to cnhuhbert.model "
                  f"(backend={backend}, mode={mode})", flush=True)


def load_tts() -> TTS:
    cfg_path = ROOT / "src" / "tts_config.yaml"
    config = TTS_Config(str(cfg_path))
    tts = TTS(config)
    _maybe_torch_compile(tts)
    return tts


def synthesize(tts: TTS, text: str) -> Iterator[tuple[int, np.ndarray]]:
    """text を合成し (sr, chunk) を順次 yield する。

    TTS_STREAMING=1 のときは streaming_mode を有効化し、合成途中のチャンクから
    順に yield する。無効時も内部仕様により 1 度だけ全体音声を yield する。
    """
    streaming = os.environ.get("TTS_STREAMING", "").lower() in ("1", "true", "yes")
    gen = tts.run({
        "text": text,
        "text_lang": "ja",
        "ref_audio_path": str(REF_AUDIO),
        "prompt_text": REF_TEXT,
        "prompt_lang": "ja",
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        # 既に文単位で渡しているので追加分割は不要
        "text_split_method": "cut0",
        "streaming_mode": streaming,
        "parallel_infer": True,
    })
    for sr, audio in gen:
        if audio is not None and len(audio) > 0:
            yield sr, audio


def play(sr: int, audio: np.ndarray) -> None:
    if audio.size == 0:
        return
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    sd.play(audio, sr, blocking=True)


class SpeechPipeline:
    """文を投入すると合成 → 順次再生してくれる 2 段ワーカー。

    speak() で文を投げる → 合成スレッドが synthesize() → 再生スレッドが play()
    再生中も次文の合成が走るので、文単位の体感レイテンシだけで聞き始められる。
    """

    _SENTINEL = object()

    def __init__(self, tts: TTS) -> None:
        self.tts = tts
        self._synth_q: queue.Queue = queue.Queue()
        self._play_q: queue.Queue = queue.Queue()
        self._synth_thread = threading.Thread(
            target=self._synth_worker, daemon=True, name="tts-synth"
        )
        self._play_thread = threading.Thread(
            target=self._play_worker, daemon=True, name="tts-play"
        )
        self._synth_thread.start()
        self._play_thread.start()

    def speak(self, sentence: str) -> None:
        if sentence and sentence.strip():
            self._synth_q.put(sentence)

    def join(self) -> None:
        """投入済みの文がすべて再生し終わるまでブロック。"""
        self._synth_q.join()
        self._play_q.join()

    def shutdown(self) -> None:
        self._synth_q.put(self._SENTINEL)
        self._synth_thread.join(timeout=5)
        self._play_q.put(self._SENTINEL)
        self._play_thread.join(timeout=5)

    def _synth_worker(self) -> None:
        while True:
            item = self._synth_q.get()
            try:
                if item is self._SENTINEL:
                    return
                try:
                    for sr, audio in synthesize(self.tts, item):
                        self._play_q.put((sr, audio))
                except Exception as e:
                    print(f"[error] TTS synth: {e}", file=sys.stderr, flush=True)
            finally:
                self._synth_q.task_done()

    def _play_worker(self) -> None:
        while True:
            item = self._play_q.get()
            try:
                if item is self._SENTINEL:
                    return
                sr, audio = item
                try:
                    play(sr, audio)
                except Exception as e:
                    print(f"[error] play: {e}", file=sys.stderr, flush=True)
            finally:
                self._play_q.task_done()


def main() -> None:
    llm = build_llm()
    print(f"[init] LLM: {type(llm).__name__} (model={llm.model})")

    print("[init] TTS を読み込み中…(初回は時間がかかります)")
    tts = load_tts()
    pipeline = SpeechPipeline(tts)
    print("[init] TTS 準備完了")

    history: list[dict] = []

    print("\n質問を入力(空行で終了):")
    try:
        while True:
            try:
                user_msg = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_msg:
                break
            history.append({"role": "user", "content": user_msg})

            buffer = ""
            full_text = ""
            try:
                for chunk in llm.chat_stream(history):
                    full_text += chunk
                    print(chunk, end="", flush=True)
                    buffer += chunk
                    sentences, buffer = split_sentences(buffer)
                    for s in sentences:
                        pipeline.speak(s)
            except Exception as e:
                print(f"\n[error] LLM: {e}", file=sys.stderr, flush=True)
                continue
            print()  # 改行
            if buffer.strip():
                pipeline.speak(buffer)
            history.append({"role": "assistant", "content": full_text})
            pipeline.join()
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()
