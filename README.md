# 藤崎 Alter Ego

## セットアップ

事前に [uv](https://github.com/astral-sh/uv) が必要です。

```sh
git clone --recurse-submodules <this repo>   # サブモジュール込みで clone
# 既に clone 済みなら: git submodule update --init --recursive
make init                                    # .venv 作成 + 依存インストール
cp .env.example .env                         # API キーを記入
```

## 起動

```sh
make             # GUI (face_chat) を起動
make voice       # CLI 版 (voice_chat) を起動
```

## 環境変数 (.env)

- `LLM_PROVIDER` — `anthropic` か `openai`
- `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL`
- `OPENAI_API_KEY` / `OPENAI_MODEL`
