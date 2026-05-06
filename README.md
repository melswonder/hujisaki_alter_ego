# 藤崎 Alter Ego

藤崎をベースにした「アルターエゴ」をフルスクリーン GUI で喋らせる音声チャット。

- LLM (Claude / ChatGPT) と会話
- 応答先頭の表情タグ `[NN]` で 16 種類の表情画像を切替
- 応答テキストを GPT-SoVITS で藤崎の声色に変換して再生

## アーキテクチャ概要

```
ユーザー入力
   │
   ▼
[LLM]  Claude or OpenAI  ──→  "[02] こんにちは、ご主人タマ♪"
   │                                │
   │                                ├─ "[02]" → 表情画像 02 (にっこり) 表示
   │                                │
   │                                └─ 残りテキスト
   │                                       │
   │                                       ▼
   │                              [GPT-SoVITS TTS]
   │                                       │
   │                                       ▼
   │                                  音声波形 → 再生
   ▼
チャットログ (画面オーバーレイ)
```

- **LLM**: 人格は `src/persona.md`。表情タグ指示は GUI 側で末尾に追加
- **TTS**: GPT-SoVITS の fine-tune 済みモデル (`huzisaki_models/`) + リファレンス音声 1 個 (`huzisaki_007.wav`) で藤崎の声を合成。**学習はもう完了済みで、毎回走るのは推論のみ**

## 必要なもの

- macOS (動作確認: darwin/arm64, Python 3.10)
- [uv](https://github.com/astral-sh/uv) (Python パッケージマネージャ)
- Anthropic API key か OpenAI API key

## セットアップ

```sh
# サブモジュール (GPT-SoVITS) ごと clone
git clone --recurse-submodules <this repo>
cd hujisaki_alter_ego

# 既に clone 済みなら
git submodule update --init --recursive

# venv 作成 + 依存インストール (GPT-SoVITS の重い依存もここで入る)
make init

# .env を作成して API キーを記入
cp .env.example .env
```

## 起動

```sh
make             # GUI 版 (フルスクリーン)
make voice       # CLI 版 (ターミナル入力)
make help        # 使えるターゲット一覧
make clean       # .venv を削除
```

GUI 版は `Esc` でフルスクリーン切替。

## 環境変数 (`.env`)

| 変数 | 役割 | 既定値 |
|------|------|--------|
| `LLM_PROVIDER` | `anthropic` か `openai` | `anthropic` |
| `ANTHROPIC_API_KEY` | Claude API キー | — |
| `ANTHROPIC_MODEL` | Claude モデル名 | `claude-sonnet-4-5` |
| `OPENAI_API_KEY` | OpenAI API キー | — |
| `OPENAI_MODEL` | OpenAI モデル名 | `gpt-4o-mini` |

## ディレクトリ構成

```
.
├── src/
│   ├── face_chat.py      # GUI (tkinter, フルスクリーン)
│   ├── voice_chat.py     # CLI + LLM/TTS 共通モジュール
│   ├── persona.md        # アルターエゴの人格プロンプト
│   └── tts_config.yaml   # GPT-SoVITS 設定
├── face/                 # 表情画像 16 枚 (AlterEgoFaceNN.png)
├── huzisaki_models/      # fine-tune 済み TTS モデル + リファレンス wav
├── GPT-SoVITS/           # 推論ライブラリ (git submodule)
├── Makefile
└── .env.example
```

## 表情番号

| ID | 表情 | ID | 表情 |
|----|------|----|------|
| 00 | 微笑み | 08 | 慌て |
| 01 | 嬉しい | 09 | 号泣 |
| 02 | にっこり | 10 | 静かに涙 |
| 03 | 驚き | 11 | 涙目 |
| 04 | 焦り | 12 | 不安 |
| 05 | ひらめき | 13 | 大笑い |
| 06 | 落ち込み | 14 | 考え中 |
| 07 | びっくり | 15 | すすり泣き |

LLM は応答の先頭に `[NN]` を付けて表情を指定 (`face_chat.py` でパースして画像切替)。

## カスタマイズ

- **声を変える**: `voice_chat.py` の `REF_AUDIO` / `REF_TEXT` を別の wav に差し替え
- **キャラを変える**: `src/persona.md` を書き換え (CLI / GUI 両方に反映)
- **モデル差し替え**: `src/tts_config.yaml` の `t2s_weights_path` / `vits_weights_path`

## トラブルシュート

- **`OverloadedError 529`**: Anthropic 側の一時過負荷。少し待つか `LLM_PROVIDER=openai` に切替
- **`ModuleNotFoundError: voice_chat`**: 必ずプロジェクトルートで `make` (= `python -m src.face_chat`) として起動。`python src/face_chat.py` 直叩きはダメ
- **TTS が遅い**: M1/M2 の MPS で FP32 推論しているため。`tts_config.yaml` の `is_half: true` で 2 倍速くなる可能性 (音質劣化リスクあり)
- **音声が出ない**: `[system] TTS 準備完了` がコンソールに出るのを待ってから入力。初回ロードは数十秒〜数分かかる
