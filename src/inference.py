from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "facial_emotion.keras"
METADATA_PATH = ROOT / "artifacts" / "metadata.json"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Download FER-2013 and run python -m src.train.")
    return keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = ImageOps.grayscale(image)
    image = ImageOps.fit(image, (48, 48), method=Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32)[np.newaxis, ..., np.newaxis]


def predict(image: Image.Image, model=None) -> dict:
    model = model or load_model()
    scores = model.predict(preprocess_image(image), verbose=0)[0]
    ranking = np.argsort(scores)[::-1]
    return {
        "predicted_label": CLASS_NAMES[int(ranking[0])],
        "confidence": float(scores[ranking[0]]),
        "probabilities": {CLASS_NAMES[i]: float(scores[i]) for i in range(len(CLASS_NAMES))},
        "top_three": [
            {"emotion": CLASS_NAMES[int(i)], "probability": float(scores[i])}
            for i in ranking[:3]
        ],
    }


def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
