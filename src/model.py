import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple

def build_data_augmentation_layers() -> tf.keras.Sequential:
    """Create data augmentation sequential pipeline to apply to training images."""
    return models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
        layers.RandomContrast(0.1)
    ], name="data_augmentation")

def build_transfer_model(
    num_classes: int,
    base_architecture: str = "EfficientNetB0",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.3
) -> tf.keras.Model:
    """
    Build transfer learning classification model with frozen base network and custom head.
    Supports EfficientNetB0 and MobileNetV2.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # Base network setup
    if base_architecture.lower() == "mobilenetv2":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
    else:
        # Default EfficientNetB0
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )

    # Freeze base model initially
    base_model.trainable = False

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"{base_architecture}_CattleBreedClassifier")
    return model

def unfreeze_model_for_finetuning(
    model: tf.keras.Model,
    unfreeze_from_layer: int = -30,
    learning_rate: float = 1e-5
) -> tf.keras.Model:
    """
    Unfreeze the top convolutional layers of the base model for fine-tuning.
    Recompiles model with a low learning rate.
    """
    # Find base model inside model layers
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "efficientnet" in layer.name.lower() or "mobilenet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        print("[WARNING] Could not locate base feature extractor model layer for unfreezing.")
        return model

    base_model.trainable = True

    # Unfreeze only layers after unfreeze_from_layer
    num_layers = len(base_model.layers)
    cutoff = max(0, num_layers + unfreeze_from_layer if unfreeze_from_layer < 0 else unfreeze_from_layer)

    for i, layer in enumerate(base_model.layers):
        if i < cutoff or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    print(f"[FINE-TUNING] Unfrozen base model layers from layer index {cutoff} to {num_layers}.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
