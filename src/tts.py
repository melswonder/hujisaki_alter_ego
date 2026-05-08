"""GPT-SoVITS TTS のロード・合成・再生パイプライン。

import 時の副作用:
  - GPT-SoVITS の ``sv.py`` が ``os.getcwd()/GPT_SoVITS/eres2net`` を
    ``sys.path`` に追加する前提なので、本モジュールは GPT-SoVITS ディレクトリへ
    chdir し、関連パスを sys.path に挿入する。
  - その後 ``GPT_SoVITS.TTS_infer_pack.TTS`` を import する。

環境変数:
  - TTS_COMPILE         1 で BERT/CN-HuBERT に torch.compile を適用
  - TTS_COMPILE_BACKEND torch.compile backend (既定: inductor)
  - TTS_COMPILE_MODE    torch.compile mode    (既定: default)
  - TTS_HALF            1 で TTS_Config.is_half を fp16 に上書き
  - TTS_STREAMING       1 で streaming_mode を有効化
"""

from __future__ import annotations

import os
import queue
import re
import sys
import threading
from collections.abc import Iterator

import numpy as np
import sounddevice as sd

from src.llm import ROOT

# ---- GPT-SoVITS インポートのための副作用 ----
os.chdir(ROOT / "GPT-SoVITS")
sys.path.insert(0, str(ROOT / "GPT-SoVITS"))
sys.path.insert(0, str(ROOT / "GPT-SoVITS" / "GPT_SoVITS"))

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

REF_AUDIO = ROOT / "huzisaki_models" / "huzisaki_007.wav"
REF_TEXT = "問題ないってことなんじゃないかな 犯人が証拠隠滅したってことじゃあ"


# ---- 文区切り ---------------------------------------------------------------

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


# ---- 起動時最適化 -----------------------------------------------------------

def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def _maybe_torch_compile(tts: TTS) -> None:
    """TTS_COMPILE=1 のとき BERT / CN-HuBERT に torch.compile をかける。

    t2s_model / vits_model は実行時に属性を差し替える箇所があり compile と
    相性が悪いので、副作用の少ない 2 つだけを対象にする。失敗してもログを
    出して続行。
    """
    if not _env_truthy("TTS_COMPILE"):
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


def _patch_sv_for_mps_half() -> None:
    """SV.compute_embedding3 が wav を fp16 にしてから Kaldi.fbank に渡すと、
    MPS の torch.fft.rfft が fp16 未対応で落ちる。fbank は fp32 で計算し、
    特徴量を half にキャストして embedding_model に渡すよう差し替える。

    TTS.py が ``from sv import SV`` で読み込むため、bare module 名 ``sv``
    をパッチする必要がある (``GPT_SoVITS.sv`` だと別モジュール扱いで効かない)。
    """
    import torch
    import sv as sv_mod
    import kaldi as Kaldi

    def compute_embedding3(self, wav):
        with torch.no_grad():
            wav_fp32 = wav.float()
            feat = torch.stack(
                [Kaldi.fbank(w.unsqueeze(0), num_mel_bins=80,
                             sample_frequency=16000, dither=0)
                 for w in wav_fp32]
            )
            if self.is_half:
                feat = feat.half()
            return self.embedding_model.forward3(feat)

    sv_mod.SV.compute_embedding3 = compute_embedding3


def load_tts() -> TTS:
    cfg_path = ROOT / "src" / "tts_config.yaml"
    config = TTS_Config(str(cfg_path))

    # env が立っていれば yaml の is_half を上書き
    if _env_truthy("TTS_HALF"):
        if str(config.device) == "cpu":
            print("[tts] TTS_HALF=1 だが device=cpu のため無視します", flush=True)
        else:
            config.is_half = True

    # yaml/env いずれの経路でも fp16 になったら MPS 用 SV パッチを当てる
    if config.is_half and str(config.device) != "cpu":
        try:
            _patch_sv_for_mps_half()
            print(f"[tts] is_half=True (fp16) で起動 (device={config.device})", flush=True)
        except Exception as e:
            print(f"[tts] SV モンキーパッチ失敗 (続行): {e}", flush=True)

    tts = TTS(config)
    _maybe_torch_compile(tts)
    return tts


# ---- 合成・再生 -------------------------------------------------------------

def synthesize(tts: TTS, text: str) -> Iterator[tuple[int, np.ndarray]]:
    """text を合成し (sr, chunk) を順次 yield する。

    TTS_STREAMING=1 のときは合成途中のチャンクから順に yield する。
    無効時も内部仕様により 1 度だけ全体音声を yield する。
    """
    gen = tts.run({
        "text": text,
        "text_lang": "ja",
        "ref_audio_path": str(REF_AUDIO),
        "prompt_text": REF_TEXT,
        "prompt_lang": "ja",
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut0",  # 文単位で渡しているので追加分割不要
        "streaming_mode": _env_truthy("TTS_STREAMING"),
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
    """文を投げると合成 → 順次再生してくれる 2 段ワーカー。

    speak() で文を投入 → 合成スレッドが synthesize() → 再生スレッドが play()。
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
