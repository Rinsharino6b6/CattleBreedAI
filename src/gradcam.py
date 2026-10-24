import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional, Union, Any

from src.config import GRADCAM_DIR, IMAGE_SIZE
from src.data_utils import load_and_preprocess_image

def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Find the name of the final 2D convolutional layer in base model or nested layers."""
    # Check for EfficientNet top_conv
    for layer in reversed(model.layers):
        if layer.name == "top_conv":
            return layer.name
        if isinstance(layer, tf.keras.Model):
            sub_conv = find_last_conv_layer(layer)
            if sub_conv:
                return f"{layer.name}/{sub_conv}"
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name

    # Check nested base model if applicable
    for layer in model.layers:
        if "efficientnet" in layer.name.lower() or "mobilenet" in layer.name.lower():
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    return sub_layer.name

    raise ValueError("No 2D convolutional layer found in model for Grad-CAM.")

def generate_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    pred_index: Optional[int] = None,
    layer_name: Optional[str] = None
) -> np.ndarray:
    """
    Generate Grad-CAM activation heatmap for input image tensor (1, 224, 224, 3).
    """
    # Locate base model or target layer
    base_model = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower() or "mobilenet" in layer.name.lower():
            base_model = layer
            break

    if base_model is not None:
        # Construct sub-gradient model inside base_model
        if layer_name is None:
            layer_name = "top_conv" if "top_conv" in [l.name for l in base_model.layers] else [l.name for l in base_model.layers if isinstance(l, tf.keras.layers.Conv2D)][-1]
        
        target_layer = base_model.get_layer(layer_name)
        grad_model = tf.keras.models.Model(
            inputs=base_model.inputs,
            outputs=[target_layer.output, base_model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            # Pass predictions through classification head layers
            x = model.get_layer("global_avg_pool")(predictions)
            if "batch_norm" in [l.name for l in model.layers]:
                x = model.get_layer("batch_norm")(x)
            logits = model.get_layer("predictions")(x)

            if pred_index is None:
                pred_index = tf.argmax(logits[0])
            class_channel = logits[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
    else:
        if layer_name is None:
            layer_name = find_last_conv_layer(model)
        
        target_layer = model.get_layer(layer_name)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

    # Compute mean intensity of gradient per feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to keep only features that positively contribute to predicted class
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_gradcam(
    original_image_path_or_pil: Union[str, Path, Any],
    heatmap: np.ndarray,
    alpha: float = 0.4,
    save_filename: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlay Grad-CAM heatmap on original image using OpenCV JET colormap.
    Returns (heatmap_rgb, overlay_rgb).
    """
    if isinstance(original_image_path_or_pil, (str, Path)):
        img = cv2.imread(str(original_image_path_or_pil))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(original_image_path_or_pil, Image.Image):
        img = np.array(original_image_path_or_pil.convert("RGB"))
    else:
        img = np.array(original_image_path_or_pil)

    img = cv2.resize(img, IMAGE_SIZE)

    # Convert heatmap to uint8 0-255 range
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_uint8 = cv2.resize(heatmap_uint8, (img.shape[1], img.shape[0]))

    # Colorize heatmap with JET colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Superimpose heatmap on original image
    overlay = cv2.addWeighted(img, 1.0 - alpha, heatmap_rgb, alpha, 0)

    if save_filename:
        save_path = GRADCAM_DIR / save_filename
        cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return heatmap_rgb, overlay
