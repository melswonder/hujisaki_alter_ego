"""Claude API → GPT-SoVITS (huzisaki voice) 音声チャット (CLI 版)。

挙動:
  - 標準入力に質問を入れる
  - LLM の応答をストリーミングで受け取り、句点 (。 ! ? ！ ？ 改行) ごとに
    1 文ずつ TTS に投入する
  - 合成は別スレッド、再生はさらに別スレッドで行うので、文 N の再生中に
    文 N+1 の合成を進められる
"""

from __future__ import annotations

import sys

from src.llm import build_llm
from src.tts import SpeechPipeline, load_tts, split_sentences


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
            print()
            if buffer.strip():
                pipeline.speak(buffer)
            history.append({"role": "assistant", "content": full_text})
            pipeline.join()
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()
