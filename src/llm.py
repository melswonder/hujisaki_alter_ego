"""LLM クライアント (Anthropic / OpenAI) と build_llm()。

このモジュールはプロジェクト全体で使う ``ROOT`` と ``SYSTEM_PROMPT`` の
単一ソースも兼ねる (重い依存を持たないので最初に import される想定)。

環境変数:
  - LLM_PROVIDER       anthropic | openai (既定: anthropic)
  - ANTHROPIC_MODEL    既定: claude-sonnet-4-5
  - OPENAI_MODEL       既定: gpt-4o-mini
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

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
            yield from stream.text_stream


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
