PYTHON_VERSION ?= 3.10
VENV ?= .venv
PY := $(VENV)/bin/python
ARGS ?=

# ANSI color codes
C_TITLE := \033[1;36m
C_CMD   := \033[1;33m
C_HEAD  := \033[1;32m
C_DIM   := \033[2;37m
C_OFF   := \033[0m

.DEFAULT_GOAL := run

.PHONY: init run eko fast streaming voice clean help

help:
	@printf "$(C_TITLE)=== alter_ego Makefile ===$(C_OFF)\n"
	@printf "$(C_CMD)make init$(C_OFF)       - submodule 取得 + .venv 作成 + 依存インストール\n"
	@printf "$(C_CMD)make / run$(C_OFF)      - GUI 起動 (ARGS で追加フラグを渡せる)\n"
	@printf "$(C_CMD)make eko$(C_OFF)        - --eko (TTS 無し)\n"
	@printf "$(C_CMD)make fast$(C_OFF)       - --fast (torch.compile reduce-overhead)\n"
	@printf "$(C_CMD)make streaming$(C_OFF)  - --streaming (TTS ストリーミング合成)\n"
	@printf "$(C_CMD)make voice$(C_OFF)      - CLI 版 (voice_chat) を起動\n"
	@printf "$(C_CMD)make clean$(C_OFF)      - .venv を削除\n"
	@printf "\n"
	@printf "$(C_HEAD)フラグは組み合わせ可能。例:$(C_OFF)\n"
	@printf "  $(C_DIM)make run ARGS=\"--eko --streaming\"$(C_OFF)\n"
	@printf "  $(C_DIM)make fast ARGS=\"--streaming\"$(C_OFF)\n"

$(VENV):
	uv venv --python $(PYTHON_VERSION) $(VENV)

init: $(VENV)
	git submodule update --init --recursive
	uv pip install --python $(PY) -r GPT-SoVITS/requirements.txt
	uv pip install --python $(PY) -r GPT-SoVITS/extra-req.txt
	uv pip install --python $(PY) anthropic openai python-dotenv sounddevice pillow
	@echo "セットアップ完了。.env を作成して API キーを設定してください (.env.example 参照)"

run:
	$(PY) -m src.face_chat $(ARGS)

eko:
	$(PY) -m src.face_chat --eko $(ARGS)

fast:
	$(PY) -m src.face_chat --fast $(ARGS)

streaming:
	$(PY) -m src.face_chat --streaming $(ARGS)

voice:
	$(PY) -m src.voice_chat

clean:
	rm -rf $(VENV)
