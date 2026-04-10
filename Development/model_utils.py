from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
from tensorflow import keras


CLASS_NAMES = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 123
LOW_CONFIDENCE_THRESHOLD = 30

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "Dataset"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "Development" / "artifacts"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / "mango_leaf_disease.keras"
DEFAULT_HISTORY_PATH = DEFAULT_ARTIFACT_DIR / "training_history.json"
DEFAULT_METRICS_PATH = DEFAULT_ARTIFACT_DIR / "evaluation.json"
DEFAULT_SPLITS_PATH = DEFAULT_ARTIFACT_DIR / "dataset_splits.json"
DEFAULT_BROWSER_MODEL_DIR = REPO_ROOT / "model"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_parent_dir(path)
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def list_dataset_images(dataset_dir: Path) -> dict[str, list[Path]]:
    dataset_dir = Path(dataset_dir)
    missing_dirs = [name for name in CLASS_NAMES if not (dataset_dir / name).is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            f"Missing class folders in {dataset_dir}: {', '.join(missing_dirs)}"
        )

    images_by_class: dict[str, list[Path]] = {}
    for class_name in CLASS_NAMES:
        class_dir = dataset_dir / class_name
        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise FileNotFoundError(f"No supported images found in {class_dir}")
        images_by_class[class_name] = image_paths
    return images_by_class


def create_dataset_splits(
    dataset_dir: Path,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, list[tuple[str, int]]], dict[str, dict[str, int]]]:
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio and validation_ratio must leave room for a test split.")

    rng = random.Random(seed)
    splits: dict[str, list[tuple[str, int]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    counts: dict[str, dict[str, int]] = {}
    images_by_class = list_dataset_images(Path(dataset_dir))

    for label_index, class_name in enumerate(CLASS_NAMES):
        image_paths = images_by_class[class_name][:]
        rng.shuffle(image_paths)

        total = len(image_paths)
        train_count = int(total * train_ratio)
        validation_count = int(total * validation_ratio)
        test_count = total - train_count - validation_count

        if min(train_count, validation_count, test_count) <= 0:
            raise ValueError(
                f"Not enough images in class '{class_name}' to create train/validation/test splits."
            )

        partitions = {
            "train": image_paths[:train_count],
            "validation": image_paths[train_count : train_count + validation_count],
            "test": image_paths[train_count + validation_count :],
        }

        counts[class_name] = {
            split_name: len(split_paths) for split_name, split_paths in partitions.items()
        }
        for split_name, split_paths in partitions.items():
            splits[split_name].extend((str(path), label_index) for path in split_paths)

    for split_name in splits:
        rng.shuffle(splits[split_name])

    return splits, counts


def load_image(
    path: tf.Tensor,
    label: tf.Tensor,
    image_size: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image_tensor = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image_tensor.set_shape([None, None, 3])
    image_tensor = tf.image.resize(image_tensor, image_size)
    image_tensor = tf.cast(image_tensor, tf.float32)
    return image_tensor, label


def build_augmenter() -> keras.Sequential:
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.12),
            keras.layers.RandomZoom(0.15),
            keras.layers.RandomTranslation(0.08, 0.08),
            keras.layers.RandomContrast(0.1),
        ],
        name="train_augmenter",
    )


def build_dataset(
    samples: Iterable[tuple[str, int]],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: tuple[int, int] = IMAGE_SIZE,
    seed: int = DEFAULT_SEED,
    training: bool = False,
    augmenter: keras.Sequential | None = None,
) -> tf.data.Dataset:
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("The requested dataset split is empty.")

    paths, labels = zip(*sample_list)
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(sample_list),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda path, label: load_image(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)

    if training and augmenter is not None:
        dataset = dataset.map(
            lambda images, labels: (augmenter(images, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.prefetch(tf.data.AUTOTUNE)


def build_model(
    *,
    class_count: int = len(CLASS_NAMES),
    image_size: tuple[int, int] = IMAGE_SIZE,
    dense_units: int = 128,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-4,
    base_weights: str = "imagenet",
) -> keras.Model:
    weights = None if str(base_weights).lower() == "none" else base_weights
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=(*image_size, 3),
        pooling="max",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*image_size, 3), name="image")
    x = base_model(inputs, training=False)
    x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(0.01),
    )(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(class_count, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mango_leaf_disease_classifier")
    compile_model(model, learning_rate=learning_rate)
    return model


def compile_model(model: keras.Model, *, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def find_backbone(model: keras.Model) -> keras.Model:
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name.startswith("efficientnetb0"):
            return layer
    raise ValueError("Could not find the EfficientNet backbone in the provided model.")


def unfreeze_for_finetuning(model: keras.Model, *, trainable_layers: int = 40) -> None:
    backbone = find_backbone(model)
    backbone.trainable = True

    freeze_until = max(0, len(backbone.layers) - trainable_layers)
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= freeze_until and not isinstance(
            layer, keras.layers.BatchNormalization
        )


def collect_targets(dataset: tf.data.Dataset) -> np.ndarray:
    return np.concatenate([labels.numpy() for _, labels in dataset], axis=0)


def evaluate_model(model: keras.Model, dataset: tf.data.Dataset) -> dict:
    metrics = model.evaluate(dataset, return_dict=True, verbose=1)
    predictions = model.predict(dataset, verbose=1)
    targets = collect_targets(dataset)
    predicted_labels = predictions.argmax(axis=1)
    confusion = tf.math.confusion_matrix(
        targets,
        predicted_labels,
        num_classes=len(CLASS_NAMES),
    ).numpy()

    per_class = {}
    for index, class_name in enumerate(CLASS_NAMES):
        support = int((targets == index).sum())
        correct = int(((targets == index) & (predicted_labels == index)).sum())
        per_class[class_name] = {
            "support": support,
            "accuracy": round(correct / support, 4) if support else 0.0,
        }

    return {
        "metrics": {key: float(value) for key, value in metrics.items()},
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def choose_source_model(model_path: str | None = None) -> Path:
    candidates = [Path(model_path)] if model_path else []
    candidates.append(DEFAULT_MODEL_PATH)
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No trained model was found. Retrain first or pass --model-path explicitly."
    )


def load_trained_model(model_path: Path, *, learning_rate: float | None = None) -> keras.Model:
    model_path = Path(model_path)
    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        if model_path.suffix.lower() not in {".h5", ".hdf5"}:
            raise

        print(f"Falling back to rebuilding the legacy H5 architecture: {exc}")
        model = build_model(
            class_count=len(CLASS_NAMES),
            image_size=IMAGE_SIZE,
            learning_rate=learning_rate or 1e-4,
            base_weights="none",
        )
        model.load_weights(model_path)

    if learning_rate is not None:
        compile_model(model, learning_rate=learning_rate)
    return model


def strip_efficientnet_preprocessing(model: keras.Model) -> keras.Model:
    backbone = find_backbone(model)
    layers = list(backbone.layers)
    if len(layers) < 4:
        return model

    if not isinstance(layers[1], keras.layers.Rescaling) or not isinstance(
        layers[2], keras.layers.Normalization
    ):
        return model

    inputs = keras.Input(shape=model.input_shape[1:], name="image")
    x = inputs

    for layer in layers[3:]:
        x = layer(x)

    for layer in model.layers[2:]:
        x = layer(x)

    browser_model = keras.Model(inputs=inputs, outputs=x, name=f"{model.name}_browser")
    browser_model.trainable = False
    return browser_model


def build_browser_metadata(
    source_model_path: Path,
    output_dir: Path,
    *,
    preprocessing: str,
) -> dict:
    return {
        "class_labels": CLASS_NAMES,
        "image_size": list(IMAGE_SIZE),
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "preprocessing": preprocessing,
        "source_model": str(source_model_path.resolve()),
        "model_dir": str(output_dir.resolve()),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }
