# ml-scratch

A scratchpad for basic ml examples, techniques, etc.


- **sklearn_to_onnx**: Example of converting an [scikit-learn](https://scikit-learn.org/stable/) model to an [onnx](https://onnx.ai) model, checking its validity, and then runing it using [onnxruntime](https://onnxruntime.ai)
- **sklearn_minilm**: An end-to-end example of building a custom sentiment analysis pipeline using [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for sentence embeddings, training a [scikit-learn](https://scikit-learn.org/stable/) classifier, exporting it to [ONNX](https://onnx.ai), and performing inference entirely using [ONNX Runtime](https://onnxruntime.ai).
