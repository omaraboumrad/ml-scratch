.phony: all sync repl clean reset

MODEL_PATH = minilm.onnx
MINILM_URL = https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx

TEXT ?= I'm not happy with the service.

all: train classify

train:
	@if [ ! -f $(MODEL_PATH) ]; then \
		echo "Downloading model..."; \
		wget https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx -O $(MODEL_PATH); \
	fi
	@uv run minilm.py train

classify:
	@uv run minilm.py classify "$(TEXT)"

sync:
	@uv sync

repl:
	@uv run ipython -i minilm.py


clean:
	@rm -rf sentiment_classifier.onnx


reset: clean
	@rm $(MODEL_PATH)
	@rm -rf .venv
