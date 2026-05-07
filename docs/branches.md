# ブランチ使い分けガイド

このリポジトリは **プラットフォームごとに長期ブランチ** を分けて運用しています。

## ブランチ構成

| ブランチ | 用途 | 主な差分 |
| --- | --- | --- |
| `main` | 共通ベース。プラットフォーム非依存のコードのみ。 | プラットフォーム固有設定は持たない。 |
| `linux` | Linux ネイティブ / WSL2 で動かす用。 | `src/tts_config.yaml` の `custom.device: cpu` |
| `windows` | Windows ネイティブ (cmd / PowerShell) で動かす用。 | `Makefile` の `PY := $(VENV)/Scripts/python.exe`、`src/tts_config.yaml` の `custom.device: cpu` |

`main` は単独で `make run` できることを保証していません。実際に動かすときは必ず `linux` か `windows` にチェックアウトしてください。

## 使い分けの判断

- WSL2 上で動かす → `linux`
- Linux 実機で動かす → `linux`
- Windows ネイティブで動かす → `windows` (GNU make が必要。chocolatey か scoop で導入)
- macOS で動かす → 現状ブランチなし。必要なら `main` から `mac` ブランチを切って `device: mps` にする

## 切り替え手順

```sh
# 初回チェックアウト
git fetch origin
git checkout linux        # もしくは windows

# 仮想環境とモデルのセットアップ (各ブランチで一度)
make init
```

`.venv` はブランチごとに作り直さなくても基本問題ないですが、Linux と Windows をひとつのワーキングツリーで往復するのは推奨しません (venv のバイナリ互換性がない)。

## 共通変更を両ブランチに反映する流れ

新機能や共通コードの修正は **必ず `main` に入れて**、そこから `linux` と `windows` に伝搬させます。

```sh
# 1. main で開発してコミット
git checkout main
# ... 編集 ...
git commit -am "feat: 共通の改善"
git push origin main

# 2. linux に取り込む
git checkout linux
git merge main
git push origin linux

# 3. windows に取り込む
git checkout windows
git merge main
git push origin windows
```

リベース運用にしたい場合は `git rebase main` でも OK ですが、リモートに既に push 済みなら force push が必要になります。基本は merge で問題ありません。

## プラットフォーム固有の修正

`linux` か `windows` だけに必要な変更は **そのブランチに直接コミット** します。`main` には入れません。

例:
- Linux のみで使うシェルスクリプト → `linux` ブランチにコミット
- Windows の PowerShell スクリプト → `windows` ブランチにコミット
- `tts_config.yaml` の `device` 値 → 各プラットフォームブランチで管理

## 新しいプラットフォームを追加するとき

```sh
git checkout main
git checkout -b mac       # 例: mac ブランチを切る
# ... プラットフォーム固有設定を編集 ...
git commit -am "mac: ..."
git push -u origin mac
```

その後、本ドキュメントの表とリストにエントリを追記してください。
