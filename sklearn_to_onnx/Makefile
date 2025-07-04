.phony: all sync repl clean reset

all:
	@uv run main.py



sync:
	@uv sync


repl:
	@uv run ipython -i main.py


clean:
	@rm -rf iris.onnx


reset: clean
	@rm -rf .venv


