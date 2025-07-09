import click
import numpy as np
import onnxruntime as ort
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from collections import Counter
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os

MODEL_PATH = "sentiment_classifier.onnx"
EMBEDDING_DIM = 384

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


@click.group()
def cli():
    pass


@cli.command("train")
def train_export():
    """Train sentiment classifier and export to ONNX"""
    print(">>> Training sentiment classifier with MiniLM ONNX <<<")
    print("Loading IMDb...")
    dataset = load_dataset("imdb", split="train").shuffle(seed=42).select(range(100))
    texts = dataset["text"]
    labels = dataset["label"]
    print("Label distribution:", Counter(labels))

    print("Tokenizing...")
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="np")

    print("Running MiniLM ONNX...")
    session = ort.InferenceSession("minilm.onnx")
    input_keys = [i.name for i in session.get_inputs()]
    inputs = {
        input_keys[0]: tokens["input_ids"],
        input_keys[1]: tokens["attention_mask"],
    }
    if "token_type_ids" in input_keys:
        inputs["token_type_ids"] = tokens.get("token_type_ids", np.zeros_like(tokens["input_ids"]))

    outputs = session.run(None, inputs)[0]
    embeddings = outputs.mean(axis=1)

    X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
        embeddings, labels, texts, test_size=0.2, random_state=42
    )

    print("Training classifier...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    print("Classification Report:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("Exporting to ONNX...")
    initial_type = [('input', FloatTensorType([None, EMBEDDING_DIM]))]
    onnx_clf = convert_sklearn(clf, initial_types=initial_type)
    with open(MODEL_PATH, "wb") as f:
        f.write(onnx_clf.SerializeToString())
    print(f"✅ Exported classifier to {MODEL_PATH}")


@cli.command("classify")
@click.argument("text")
def classify(text):
    """Run ONNX-only inference on input text"""
    if not os.path.exists(MODEL_PATH):
        click.echo("❌ You must run `train` first.")
        return

    print(f"Classifying: {text}")
    tokenized = tokenizer([text], padding=True, truncation=True, return_tensors="np")

    session_embed = ort.InferenceSession("minilm.onnx")
    session_clf = ort.InferenceSession(MODEL_PATH)

    input_map = {
        session_embed.get_inputs()[0].name: tokenized["input_ids"],
        session_embed.get_inputs()[1].name: tokenized["attention_mask"],
    }
    if "token_type_ids" in [i.name for i in session_embed.get_inputs()]:
        input_map["token_type_ids"] = tokenized.get("token_type_ids", np.zeros_like(tokenized["input_ids"]))

    embedding = session_embed.run(None, input_map)[0].mean(axis=1).astype(np.float32)

    input_name = session_clf.get_inputs()[0].name
    prediction = session_clf.run(None, {input_name: embedding})[0]

    label = int(prediction[0])
    sentiment = "POSITIVE" if label == 1 else "NEGATIVE"
    print(f"Prediction: {label} ({sentiment})")


if __name__ == "__main__":
    cli()

