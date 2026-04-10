from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import tf_keras
from tensorflow import keras

from model_utils import (
    CLASS_NAMES,
    DEFAULT_BROWSER_MODEL_DIR,
    IMAGE_SIZE,
    build_browser_metadata,
    choose_source_model,
    load_trained_model,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the trained Keras model into TensorFlow.js format for GitHub Pages."
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Path to the source .keras or .h5 model. Defaults to the newest local training output.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_BROWSER_MODEL_DIR))
    return parser.parse_args()


def clear_previous_export(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("model.json", "metadata.json", "group*-shard*of*"):
        for path in output_dir.glob(pattern):
            path.unlink()


def load_tfjs_keras_exporter():
    spec = importlib.util.find_spec("tensorflowjs")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(
            "tensorflowjs is not installed. Install the packages from requirements.txt first."
        )

    package_dir = Path(spec.submodule_search_locations[0])
    converters_dir = package_dir / "converters"

    for module_name in list(sys.modules):
        if module_name == "tensorflowjs" or module_name.startswith("tensorflowjs."):
            sys.modules.pop(module_name, None)

    tfjs_package = types.ModuleType("tensorflowjs")
    tfjs_package.__path__ = [str(package_dir)]
    sys.modules["tensorflowjs"] = tfjs_package

    converters_package = types.ModuleType("tensorflowjs.converters")
    converters_package.__path__ = [str(converters_dir)]
    sys.modules["tensorflowjs.converters"] = converters_package

    def load_module(module_name: str, module_path: Path):
        module_spec = importlib.util.spec_from_file_location(module_name, module_path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Could not load {module_name} from {module_path}")

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
        return module

    version_module = load_module("tensorflowjs.version", package_dir / "version.py")
    quantization_module = load_module("tensorflowjs.quantization", package_dir / "quantization.py")
    read_weights_module = load_module("tensorflowjs.read_weights", package_dir / "read_weights.py")
    write_weights_module = load_module("tensorflowjs.write_weights", package_dir / "write_weights.py")
    common_module = load_module(
        "tensorflowjs.converters.common",
        converters_dir / "common.py",
    )
    keras_h5_conversion_module = load_module(
        "tensorflowjs.converters.keras_h5_conversion",
        converters_dir / "keras_h5_conversion.py",
    )

    tfjs_package.version = version_module
    tfjs_package.quantization = quantization_module
    tfjs_package.read_weights = read_weights_module
    tfjs_package.write_weights = write_weights_module
    converters_package.common = common_module
    converters_package.keras_h5_conversion = keras_h5_conversion_module

    return keras_h5_conversion_module


def build_tfjs_compatible_model(source_model: keras.Model) -> tf_keras.Model:
    base = tf_keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(*IMAGE_SIZE, 3),
        pooling="max",
    )
    base.trainable = False

    inputs = tf_keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = base(inputs, training=False)
    x = tf_keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = tf_keras.layers.Dense(
        128,
        activation="relu",
    )(x)
    x = tf_keras.layers.Dropout(0.3)(x)
    outputs = tf_keras.layers.Dense(
        len(CLASS_NAMES),
        activation="softmax",
        name="predictions",
    )(x)

    export_model = tf_keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="mango_leaf_disease_classifier",
    )
    export_model.set_weights(source_model.get_weights())
    export_model.trainable = False
    return export_model


def build_tfjs_browser_model(source_model: keras.Model) -> tf_keras.Model:
    compatible_model = build_tfjs_compatible_model(source_model)
    compatible_backbone = compatible_model.layers[1]

    stripped_backbone = tf_keras.Model(
        inputs=compatible_backbone.layers[3].input,
        outputs=compatible_backbone.output,
        name="efficientnetb0_browser",
    )
    stripped_backbone.trainable = False

    inputs = tf_keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = stripped_backbone(inputs, training=False)
    x = tf_keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = tf_keras.layers.Dense(
        128,
        activation="relu",
    )(x)
    x = tf_keras.layers.Dropout(0.3)(x)
    outputs = tf_keras.layers.Dense(
        len(CLASS_NAMES),
        activation="softmax",
        name="predictions",
    )(x)

    browser_model = tf_keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="mango_leaf_disease_classifier",
    )

    browser_model.layers[1].set_weights(stripped_backbone.get_weights())
    browser_model.layers[2].set_weights(compatible_model.layers[2].get_weights())
    browser_model.layers[3].set_weights(compatible_model.layers[3].get_weights())
    browser_model.layers[4].set_weights(compatible_model.layers[4].get_weights())
    browser_model.layers[5].set_weights(compatible_model.layers[5].get_weights())
    browser_model.trainable = False
    return browser_model


def main() -> None:
    keras_h5_conversion = load_tfjs_keras_exporter()
    save_keras_model = keras_h5_conversion.save_keras_model

    args = parse_args()
    source_model_path = choose_source_model(args.model_path or None)
    output_dir = Path(args.output_dir)

    print(f"Loading model from {source_model_path}")
    source_model = load_trained_model(source_model_path)
    preprocessing = "efficientnet_rgb_mean_std"
    export_model = build_tfjs_browser_model(source_model)
    metadata = build_browser_metadata(
        source_model_path,
        output_dir,
        preprocessing=preprocessing,
    )

    clear_previous_export(output_dir)
    save_keras_model(
        export_model,
        str(output_dir),
        metadata=metadata,
    )
    write_json(output_dir / "metadata.json", metadata)

    print(f"Exported TensorFlow.js model to {output_dir}")
    print(f"Wrote metadata to {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
