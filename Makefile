PYTHON_VERSION ?= 3.10
VENV ?= .venv
PY := $(VENV)/Scripts/python.exe

.DEFAULT_GOAL := run

.PHONY: init run voice clean help

help:
	@echo "make init   - submodule 取得 + .venv 作成 + 依存インストール"
	@echo "make / run  - GUI (face_chat) を起動"
	@echo "make voice  - CLI 版 (voice_chat) を起動"
	@echo "make clean  - .venv を削除"

$(VENV):
	uv venv --python $(PYTHON_VERSION) $(VENV)

init: $(VENV)
	git submodule update --init --recursive
	uv pip install --python $(PY) -r GPT-SoVITS/requirements.txt
	uv pip install --python $(PY) -r GPT-SoVITS/extra-req.txt
	uv pip install --python $(PY) anthropic openai python-dotenv sounddevice pillow
	@echo "セットアップ完了。.env を作成して API キーを設定してください (.env.example 参照)"

run:
	$(PY) -m src.face_chat

voice:
	$(PY) -m src.voice_chat

clean:
	rm -rf $(VENV)
