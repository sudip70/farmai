from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from model_utils import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET_DIR,
    DEFAULT_HISTORY_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SEED,
    DEFAULT_SPLITS_PATH,
    IMAGE_SIZE,
    build_augmenter,
    build_dataset,
    build_model,
    create_dataset_splits,
    ensure_parent_dir,
    evaluate_model,
    load_trained_model,
    unfreeze_for_finetuning,
    write_json,
)


def parse_args() -> argparse.Namespace:
    default_resume_model = ""
    if DEFAULT_MODEL_PATH.exists():
        default_resume_model = str(DEFAULT_MODEL_PATH)

    parser = argparse.ArgumentParser(
        description="Retrain or fine-tune the mango leaf disease classifier using the local Dataset folder."
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--resume-model",
        default=default_resume_model,
        help="Existing .keras or .h5 model to continue training from. Defaults to the latest local artifact when present. Use 'none' to start fresh.",
    )
    parser.add_argument("--history-path", default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH))
    parser.add_argument("--splits-path", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=4)
    parser.add_argument("--fine-tune-layers", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--base-weights",
        default="imagenet",
        help="Backbone weights to use when starting fresh. Use 'none' for a fully offline start.",
    )
    return parser.parse_args()


def load_or_build_model(args: argparse.Namespace) -> keras.Model:
    resume_model = args.resume_model.strip()
    if resume_model and resume_model.lower() != "none":
        model_path = Path(resume_model)
        if not model_path.exists():
            raise FileNotFoundError(f"Could not find resume model: {model_path}")
        model = load_trained_model(model_path, learning_rate=args.learning_rate)
        print(f"Continuing training from {model_path}")
        return model

    print("Starting from a fresh EfficientNetB0 backbone")
    return build_model(
        class_count=len(CLASS_NAMES),
        image_size=IMAGE_SIZE,
        learning_rate=args.learning_rate,
        base_weights=args.base_weights,
    )


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    output_model = Path(args.output_model)
    history_path = Path(args.history_path)
    metrics_path = Path(args.metrics_path)
    splits_path = Path(args.splits_path)

    splits, split_counts = create_dataset_splits(
        dataset_dir=dataset_dir,
        train_ratio=0.7,
        validation_ratio=0.15,
        seed=args.seed,
    )

    augmenter = build_augmenter()
    train_dataset = build_dataset(
        splits["train"],
        batch_size=args.batch_size,
        image_size=IMAGE_SIZE,
        seed=args.seed,
        training=True,
        augmenter=augmenter,
    )
    validation_dataset = build_dataset(
        splits["validation"],
        batch_size=args.batch_size,
        image_size=IMAGE_SIZE,
    )
    test_dataset = build_dataset(
        splits["test"],
        batch_size=args.batch_size,
        image_size=IMAGE_SIZE,
    )

    model = load_or_build_model(args)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("Training the classification head")
    training_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    history_payload = {
        "head_training": {
            key: [float(value) for value in values]
            for key, values in training_history.history.items()
        }
    }

    if args.fine_tune_epochs > 0:
        print("Fine-tuning the top EfficientNet layers")
        unfreeze_for_finetuning(model, trainable_layers=args.fine_tune_layers)
        compile_model(model, learning_rate=args.fine_tune_learning_rate)
        fine_tune_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
            verbose=1,
        )
        history_payload["fine_tuning"] = {
            key: [float(value) for value in values]
            for key, values in fine_tune_history.history.items()
        }

    evaluation = evaluate_model(model, test_dataset)
    evaluation["dataset"] = {
        "dataset_dir": str(dataset_dir.resolve()),
        "class_labels": CLASS_NAMES,
        "split_counts": split_counts,
        "seed": args.seed,
    }

    ensure_parent_dir(output_model)
    model.save(output_model)
    write_json(history_path, history_payload)
    write_json(metrics_path, evaluation)
    write_json(
        splits_path,
        {
            "dataset_dir": str(dataset_dir.resolve()),
            "class_labels": CLASS_NAMES,
            "seed": args.seed,
            "split_counts": split_counts,
        },
    )

    print(f"Saved trained model to {output_model}")
    print(f"Saved training history to {history_path}")
    print(f"Saved evaluation summary to {metrics_path}")


if __name__ == "__main__":
    main()
