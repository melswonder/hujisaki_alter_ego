# hujisaki Alter Ego

※GPUを積んでないとほぼ動かないですmacのM2で推論に3分ほどかかります

<img width="1470" height="920" alt="スクリーンショット 2026-05-08 2 05 49" src="https://github.com/user-attachments/assets/f3588188-24cf-46bc-ad29-0905604eaeca" />

ダンガンロンパに登場する「アルターエゴ」GUI で喋らせる音声チャット。

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

#サブモジュール + venv 作成 + 依存インストール (GPT-SoVITS の重い依存もここで入る)
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

## 使用技術詳細

### 1. LLM レイヤ

| 役割 | 採用技術 |
|------|----------|
| 会話モデル | Anthropic Claude (`claude-sonnet-4-5`) / OpenAI ChatGPT (`gpt-4o-mini`) |
| SDK | `anthropic` 0.99.0 / `openai` 2.34.0 |
| プロバイダ抽象化 | `LLMClient` 基底クラス + `build_llm()` で `LLM_PROVIDER` 環境変数によって切替 (`src/voice_chat.py`) |
| 人格設計 | `src/persona.md` をシステムプロンプトとして注入。GUI ではさらに表情タグ指示 (`[NN]` を先頭に強制) を末尾に追記 |
| 履歴管理 | プロセスメモリ上の `list[dict]` (role/content)。永続化はしない。ユーザーが終了すれば履歴は消える |
| 安全装置 | `max_tokens=1024` で長すぎる応答を抑止。529 過負荷時はフォールバック無しで例外伝播 (UI で `[error]` 表示) |

### 2. TTS レイヤ — GPT-SoVITS

GPT-SoVITS は **2 段ニューラルネット** + **few-shot リファレンス条件付け** という構造の音声合成器。

#### 2.1 アーキテクチャ

```
入力テキスト ──► [GPT (T2S)] ──► 音響セマンティックトークン ──► [SoVITS (S2W)] ──► 波形 (32kHz)
                    ▲                                                ▲
                    │                                                │
              REF_TEXT (ref の文)                            REF_AUDIO (5〜10秒の wav)
                                                                     │
                                                              ┌──────┴──────┐
                                                              │ HuBERT 抽出  │  cnhuhbert_base_path
                                                              │ BERT 埋込   │  bert_base_path
                                                              └─────────────┘
```

| 段 | 中身 | チェックポイント |
|----|------|------------------|
| **GPT (T2S = Text-to-Semantic)** | 自己回帰型 Transformer (GPT-likely)。テキスト + リファレンス音響トークン → 続きの音響トークンを 1 個ずつ予測 | `huzisaki_models/huzisaki-e15.ckpt` (15 エポック fine-tune) |
| **SoVITS (S2W = Semantic-to-Wave)** | VITS ベース (Variational Inference + Adversarial + Flow)。音響トークン + 話者埋込 → mel spectrogram → vocoder で波形 | `huzisaki_models/huzisaki_e8_s192.pth` (8 エポック fine-tune, 192 ステップ) |
| **テキスト埋込 (BERT)** | `chinese-roberta-wwm-ext-large` を内部で利用。日本語含むトークンの意味埋込 | submodule の `pretrained_models/` |
| **音響埋込 (HuBERT)** | `chinese-hubert-base` で REF_AUDIO の音響特徴を抽出 | submodule の `pretrained_models/` |

VITS の中身: encoder で潜在変数の事後分布を出し、normalizing flow で整形、adversarial loss で discriminator と競争しながら mel→波形。

#### 2.2 学習 vs 推論の役割分担

- **学習 (もう完了済み)**: 藤崎の音声データで GPT 部と SoVITS 部の重みを fine-tune。`huzisaki-e15.ckpt` と `huzisaki_e8_s192.pth` がその成果物。**実行時には走らない**
- **推論 (毎回走るのはこっち)**: 学習済み重み + リファレンス wav 1 個 (`huzisaki_007.wav`) + そのテキストを毎回読み込んで forward 計算

リファレンス wav は「**今回の話し方の見本**」として動作 (zero-shot voice cloning)。学習で焼き込まれた声質ベース + リファレンスでその回の抑揚・テンションを決める、というハイブリッド。

#### 2.3 推論パラメータ (`src/voice_chat.py:46-57`)

| パラメータ | 値 | 役割 |
|-----------|----|------|
| `top_k` | 15 | サンプリング時の上位 K 個から選ぶ |
| `top_p` | 1.0 | nucleus sampling 無効 (top_k のみ使用) |
| `temperature` | 1.0 | 確率分布のシャープネス。1 で素直 |
| `text_split_method` | `cut5` | 長文を句読点で分割して順次推論 (メモリ節約) |
| `streaming_mode` | `False` | 全部生成してから一括再生 |
| `parallel_infer` | `True` | 分割した chunk を並列推論 |

#### 2.4 ハードウェア設定 (`src/tts_config.yaml`)

- `device: mps` — Apple Silicon の Metal Performance Shaders (M1/M2/M3 GPU)
- `is_half: false` — FP32 精度。MPS の FP16 は不安定なケースがあるため

CUDA NVIDIA GPU マシンなら `device: cuda` + `is_half: true` で 5〜10 倍速くなる。

### 3. オーディオ I/O

| 用途 | ライブラリ |
|------|-----------|
| 波形再生 | `sounddevice` 0.5.5 (PortAudio バインディング) |
| 数値配列 | `numpy` 1.26.4 (`<2.0` は GPT-SoVITS の制約) |
| 波形操作 | `librosa` 0.10.2 (内部で resampling 等) |
| エンコード | `torchaudio` 2.11.0 |

サンプルレート 32kHz で出力。`np.float32` に正規化してから `sounddevice.play(blocking=True)` で再生。

### 4. GUI レイヤ

| 役割 | 技術 |
|------|------|
| ウィンドウ | `tkinter` (Tcl/Tk 9.0 同梱) |
| 画像処理 | `Pillow` 10.4.0 (PNG 読み込み + LANCZOS リサンプリング + crop) |
| レイアウト | `place()` で絶対座標オーバーレイ |
| 透過レンダリング | 単一 `Canvas` 上で `create_image` (顔) と `create_text` (チャットログ) を同居させ、テキストをウィジェット背景なしで画像に直接描画 |
| 縁取り (text shadow) | 同じテキストを 4 方向 (±1px) に黒で描画してから前景文字を上書き |
| フルスクリーン | `wm_attributes('-fullscreen', True)`、Esc で切替 |
| 非ブロック化 | `threading.Thread` で LLM/TTS を別スレッド、`queue.Queue` 経由で UI スレッドへ結果を戻す。50ms 間隔で `root.after` ポーリング |
| 表情タグパース | `^\s*[\[［](\d{1,2})[\]］]\s*` の正規表現で半角/全角ブラケット両対応 |

**tkinter の制約**: ウィジェット単位の真の透過は基本的に未サポート。Entry はウィジェット背景色を持たざるを得ないため、`bg=#0d0d0d` の暗色 + 細枠で「画面に溶け込む」風に妥協。日本語 IME を捨てれば Canvas ベースの自前 Entry で完全透過は可能だが、それは犠牲が大きいので採用せず。

### 5. パッケージ管理 / ビルド

| 役割 | 技術 |
|------|------|
| Python ランタイム | CPython 3.10 (uv が `~/.local/share/uv/python` に管理) |
| 仮想環境 | `uv venv` (`.venv/` 直下) |
| 依存解決 | `uv pip install` (Rust 実装で pip より大幅に高速) |
| タスクランナー | GNU make (`Makefile`) |
| サブモジュール | `git submodule` で GPT-SoVITS を pin 留め (commit `08d627c3`) |
| 環境変数ロード | `python-dotenv` 1.2.2 (`.env` を `os.environ` に注入) |

### 6. プロセスフロー (1 ターン)

```
User Enter
  │
  ├─► entry.delete + on_send()
  │
  ├─► 履歴に user メッセージ追加
  │
  └─► 別スレッドで LLMClient.chat()
            │
            ├─ Anthropic.messages.create or OpenAI.chat.completions.create
            │
            └─► result_q.put(("reply", text))
                       │
                       ▼
                 main thread (50ms poll)
                       │
                       ├─ parse_face() で [NN] 抽出
                       ├─ Canvas image 切替 (cover scale)
                       ├─ Canvas text 追加 (3行間隔)
                       └─► 別スレッドで synthesize() + play()
                                │
                                ├─ TTS.run() で波形生成 (重い)
                                └─ sounddevice.play(blocking=True)
```

### 7. 主要バージョン (動作確認済み)

```
Python      3.10.20
torch       2.11.0
torchaudio  2.11.0
transformers 4.50.0
numpy       1.26.4
anthropic   0.99.0
openai      2.34.0
Pillow      10.4.0
sounddevice 0.5.5
Tcl/Tk      9.0
```

## トラブルシュート

- **`OverloadedError 529`**: Anthropic 側の一時過負荷。少し待つか `LLM_PROVIDER=openai` に切替
- **`ModuleNotFoundError: voice_chat`**: 必ずプロジェクトルートで `make` (= `python -m src.face_chat`) として起動。`python src/face_chat.py` 直叩きはダメ
- **TTS が遅い**: M1/M2 の MPS で FP32 推論しているため。`tts_config.yaml` の `is_half: true` で 2 倍速くなる可能性 (音質劣化リスクあり)
- **音声が出ない**: `[system] TTS 準備完了` がコンソールに出るのを待ってから入力。初回ロードは数十秒〜数分かかる

## 著作権・利用について

本プロジェクトは **個人の学習・技術検証を目的とした非営利の制作物** です。商用利用・配布・再頒布は想定していません。

- **キャラクター「藤崎」「アルターエゴ」および表情画像 (`face/`)**: ゲーム『ダンガンロンパ』シリーズに登場するキャラクターをモチーフにしています。キャラクター・画像の著作権はすべて **株式会社スパイク・チュンソフト** および各権利者に帰属します。**本リポジトリには同梱しません** (各自学習目的の範囲内でご用意ください)。本リポジトリは公式とは一切関係ありません
- **音声モデル (`huzisaki_models/`) / リファレンス音声**: 原作の音声を素材として fine-tune した派生物であり、元音声の著作権は権利者に帰属します。**本リポジトリにモデル重みやリファレンス音声は同梱しません** (各自学習目的の範囲内でご用意ください)
- **GPT-SoVITS**: 別ライセンスのサブモジュールです。詳細は `GPT-SoVITS/` 内のライセンス表記を参照してください
- **本リポジトリのソースコード部分** (`src/`, `Makefile` 等): 学習目的での参照・改変は自由ですが、上記の権利物と組み合わせた状態での再配布は行わないでください

権利者からの削除要請があった場合は速やかに対応します。あくまで **音声合成・LLM 連携・GUI 統合の技術学習を目的とした個人習作** であり、原作・公式コンテンツの代替や毀損を意図するものではありません。
