# sklearn_minilm

An end-to-end example of building a **sentiment analysis pipeline** using [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for sentence embeddings, training a [scikit-learn](https://scikit-learn.org/stable/) classifier, exporting it to [ONNX](https://onnx.ai), and performing inference entirely using [ONNX Runtime](https://onnxruntime.ai).

## Features

- Loads and preprocesses the [IMDb dataset](https://huggingface.co/datasets/imdb) using `datasets`
- Generates sentence embeddings using a pre-trained MiniLM ONNX model
- Trains a `LogisticRegression` classifier on top of MiniLM embeddings
- Exports the classifier to ONNX using `skl2onnx`
- Performs fast, ONNX-only inference using `onnxruntime`
- Clean CLI interface powered by [`click`](https://palletsprojects.com/p/click/) and `make`

## Usage

```bash
# Install and run everything
$ make

# Only train
$ make train

# Only classify
$ make classify

# Classify a single sentence
$ make classify TEXT="This is a great movie!"

# Run with uv
$ uv run minilm train
$ uv run minilm classify "This is a great movie!"
```

## Notes

- MiniLM is about 90mb
- The dataset is only partially used (100 samples) to run locally fast (Dataset is close to 100MB)
- Finally trained model is tiny in size (less than 1MB) with an accuracy of 70% on the test set (This gets better if we're using more data)
