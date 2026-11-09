from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "fer2013"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "facial_emotion.keras"
METADATA_PATH = ARTIFACTS / "metadata.json"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMAGE_SIZE = (48, 48)
EXPERIMENT_NAME = "facial-emotion-recognition"


def resolve_split(name: str) -> Path:
    candidates = [DATA_DIR / name, DATA_DIR / name.capitalize()]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Expected '{name}' directory inside {DATA_DIR}")


def validate_dataset() -> tuple[Path, Path]:
    train_dir, test_dir = resolve_split("train"), resolve_split("test")
    for split in (train_dir, test_dir):
        actual = {path.name.lower() for path in split.iterdir() if path.is_dir()}
        missing = set(CLASS_NAMES) - actual
        if missing:
            raise ValueError(f"{split} is missing class folders: {sorted(missing)}")
    return train_dir, test_dir


def build_datasets(batch_size: int):
    train_dir, test_dir = validate_dataset()
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir, labels="inferred", label_mode="int", color_mode="grayscale",
        class_names=CLASS_NAMES, image_size=IMAGE_SIZE, batch_size=batch_size,
        validation_split=0.15, subset="training", seed=42, shuffle=True,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        train_dir, labels="inferred", label_mode="int", color_mode="grayscale",
        class_names=CLASS_NAMES, image_size=IMAGE_SIZE, batch_size=batch_size,
        validation_split=0.15, subset="validation", seed=42, shuffle=True,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir, labels="inferred", label_mode="int", color_mode="grayscale",
        class_names=CLASS_NAMES, image_size=IMAGE_SIZE, batch_size=batch_size,
        shuffle=False,
    )
    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
        test_ds.prefetch(autotune),
    )


def build_model(variant: str) -> keras.Model:
    inputs = keras.Input(shape=(48, 48, 1), name="face")
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.04)(x)
    widths = [32, 64, 128] if variant == "cnn_v1" else [32, 64, 128, 192]
    for index, width in enumerate(widths):
        x = layers.Conv2D(width, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(width, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.15 + index * 0.05)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="emotion_probabilities")(x)
    model = keras.Model(inputs, outputs, name=variant)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(epochs: int = 30, batch_size: int = 64) -> dict:
    train_ds, val_ds, test_ds = build_datasets(batch_size)
    labels = np.concatenate([y.numpy() for _, y in train_ds.unbatch().batch(4096)])
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(CLASS_NAMES)), y=labels)
    class_weights = {int(i): float(weight) for i, weight in enumerate(weights)}

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri()))
    mlflow.set_experiment(EXPERIMENT_NAME)
    candidates = []

    for variant in ("cnn_v1", "cnn_v2"):
        with mlflow.start_run(run_name=variant) as run:
            model = build_model(variant)
            history = model.fit(
                train_ds, validation_data=val_ds, epochs=epochs,
                class_weight=class_weights,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
                ],
                verbose=2,
            )
            val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
            test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
            mlflow.log_params({
                "architecture": variant, "epochs_requested": epochs,
                "batch_size": batch_size, "optimizer": "Adam",
                "learning_rate": 1e-3, "image_size": "48x48 grayscale",
                "class_weighting": "balanced", "augmentation": "horizontal_flip_rotation",
            })
            mlflow.log_metrics({
                "val_loss": float(val_loss), "val_accuracy": float(val_accuracy),
                "test_loss": float(test_loss), "test_accuracy": float(test_accuracy),
                "epochs_completed": len(history.history["loss"]),
            })
            mlflow.tensorflow.log_model(model, artifact_path="model")
            candidates.append({
                "architecture": variant, "run_id": run.info.run_id,
                "val_accuracy": float(val_accuracy), "val_loss": float(val_loss),
                "test_accuracy": float(test_accuracy), "test_loss": float(test_loss),
            })

    selected = max(candidates, key=lambda item: item["val_accuracy"])
    best_model = mlflow.tensorflow.load_model(f"runs:/{selected['run_id']}/model")
    best_model.save(MODEL_PATH)
    metadata = {
        "dataset": "FER-2013 (Kaggle msambare/fer2013)",
        "classes": CLASS_NAMES,
        "input_shape": [48, 48, 1],
        "selected_model": selected["architecture"],
        "selection_metric": "validation_accuracy",
        "candidates": candidates,
        "limitations": "FER-2013 labels are noisy and class-imbalanced. Predictions are coarse dataset categories, not reliable measurements of a person's actual emotional state.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    train(args.epochs, args.batch_size)
