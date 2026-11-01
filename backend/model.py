"""
Digit Classification Model Definition and Image Preprocessing Pipeline
"""

import io
import base64
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps

logger = logging.getLogger("digit_recognition")


class DigitMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for Handwritten Digit Recognition.
    Architecture:
      Flatten (28x28 -> 784)
      -> Linear(784, 128) -> ReLU
      -> Linear(128, 64) -> ReLU
      -> Linear(64, 10)
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ModelManager:
    """Manages loading the PyTorch model and executing CPU-optimized inference."""

    def __init__(self, model_path: str = None):
        self.device = torch.device("cpu")
        self.model_path = self._resolve_model_path(model_path)
        self.model: DigitMLP = None
        self.metadata: Dict[str, Any] = {}
        self.load_model()

    def _resolve_model_path(self, model_path: str = None) -> Path:
        """Resolve model path using portable relative lookups."""
        if model_path:
            p = Path(model_path)
            if p.exists():
                return p.resolve()

        # Check relative to backend/ directory
        base_dir = Path(__file__).resolve().parent.parent
        candidates = [
            base_dir / "models" / "best_digit_model.pth",
            Path("models/best_digit_model.pth"),
            Path("best_digit_model.pth"),
        ]

        for cand in candidates:
            if cand.exists():
                return cand.resolve()

        raise FileNotFoundError(
            f"Could not locate 'best_digit_model.pth' in any standard location: "
            f"{[str(c) for c in candidates]}"
        )

    def load_model(self):
        """Loads model weights from checkpoint and sets eval mode."""
        logger.info(f"Loading checkpoint from: {self.model_path}")
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )

        self.model = DigitMLP().to(self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            self.metadata = {
                "accuracy": checkpoint.get("accuracy"),
                "learning_rate": checkpoint.get("learning_rate"),
                "batch_size": checkpoint.get("batch_size"),
                "epochs": checkpoint.get("epochs"),
            }
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise TypeError(f"Unexpected checkpoint structure: {type(checkpoint)}")

        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info("Model weights successfully loaded and set to eval mode.")

    def is_ready(self) -> bool:
        return self.model is not None


def decode_image(image_input: str) -> Image.Image:
    """
    Decodes an image from base64 data string (with or without data:image prefix).
    """
    if "," in image_input:
        header, image_input = image_input.split(",", 1)

    try:
        image_bytes = base64.b64decode(image_input)
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        raise ValueError(f"Invalid image format or failed base64 decoding: {str(e)}")


def image_to_base64(img: Image.Image) -> str:
    """Encodes a PIL Image into base64 PNG string."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def preprocess_image(image: Image.Image) -> Tuple[torch.Tensor, str, Dict[str, Any]]:
    """
    Preprocess drawn image to match standard MNIST specifications:
      1. Original Image capture.
      2. Color polarity check (white stroke on black background).
      3. Bounding-box detection & Cropping.
      4. Aspect-ratio preserving scaling to fit within a 20x20 box.
      5. Two-step Center of Mass translation:
         - Step 5a: Coarse placement near (13.5, 13.5).
         - Step 5b: Sub-pixel affine translation to align Center of Mass to exact (13.50, 13.50).
         - Clamped to ensure the digit remains strictly inside the [0, 27] boundary.
      6. Pixel normalization to [0.0, 1.0].
      7. Conversion to PyTorch Tensor.

    Returns:
      tensor: PyTorch tensor of shape (1, 1, 28, 28)
      preview_base64: PNG base64 string of final 28x28 image
      debug_info: Comprehensive debugging data for pipeline inspection
    """
    # 1. Original image capture (convert to RGB for display)
    orig_display = image.convert("RGB")
    original_b64 = image_to_base64(orig_display)

    # Convert RGBA / RGB to RGBA first to inspect transparency
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # If image has transparent background, replace transparency with black
    bg = Image.new("RGBA", image.size, (0, 0, 0, 255))
    bg.paste(image, (0, 0), image)
    gray = bg.convert("L")

    arr = np.array(gray, dtype=np.float32)

    # Check corners to auto-detect inverted drawings (e.g. black ink on white canvas)
    h, w = arr.shape
    corner_samples = [
        arr[0:5, 0:5],
        arr[0:5, w - 5 : w],
        arr[h - 5 : h, 0:5],
        arr[h - 5 : h, w - 5 : w],
    ]
    avg_corner_intensity = np.mean([np.mean(c) for c in corner_samples])

    # If background is bright, invert so stroke is white (255) and background is black (0)
    if avg_corner_intensity > 127:
        gray = ImageOps.invert(gray)
        arr = np.array(gray, dtype=np.float32)

    # 2. Bounding Box Detection
    threshold = 20.0
    active_rows = np.where(arr > threshold)[0]
    active_cols = np.where(arr > threshold)[1]

    if len(active_rows) == 0 or len(active_cols) == 0:
        raise ValueError("Canvas is empty. Please draw a digit before predicting.")

    ymin, ymax = int(active_rows.min()), int(active_rows.max())
    xmin, xmax = int(active_cols.min()), int(active_cols.max())

    box_w = xmax - xmin + 1
    box_h = ymax - ymin + 1

    # Require minimum size (filters out accidental tiny clicks or single speckles)
    if box_w < 3 and box_h < 3:
        raise ValueError("Drawing is too small or faint. Please draw a clearer digit.")

    # 3. Crop Bounding Box
    cropped_img = gray.crop((xmin, ymin, xmax + 1, ymax + 1))
    cropped_b64 = image_to_base64(cropped_img)

    # 4. Aspect-ratio preserving resize to fit ~20x20
    if box_w > box_h:
        new_w = 20
        new_h = max(1, int(round((box_h * 20.0) / box_w)))
    else:
        new_h = 20
        new_w = max(1, int(round((box_w * 20.0) / box_h)))

    resized_img = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_b64 = image_to_base64(resized_img)

    # 5. Two-Step Center of Mass Translation into 28x28 Canvas
    res_arr = np.array(resized_img, dtype=np.float32)
    tot_mass = res_arr.sum()
    if tot_mass > 0:
        y_indices, x_indices = np.indices(res_arr.shape)
        orig_cy = float(np.sum(y_indices * res_arr) / tot_mass)
        orig_cx = float(np.sum(x_indices * res_arr) / tot_mass)
    else:
        orig_cy = new_h / 2.0
        orig_cx = new_w / 2.0

    # Step 5a: Coarse placement near center (13.5, 13.5)
    paste_x = int(round(13.5 - orig_cx))
    paste_y = int(round(13.5 - orig_cy))

    # Clamp paste so resized box never exceeds canvas initially
    paste_x = max(0, min(28 - new_w, paste_x))
    paste_y = max(0, min(28 - new_h, paste_y))

    canvas_intermediate = Image.new("L", (28, 28), 0)
    canvas_intermediate.paste(resized_img, (paste_x, paste_y))

    # Step 5b: Sub-pixel translation to target (13.50, 13.50) exactly
    arr_intermediate = np.array(canvas_intermediate, dtype=np.float32)
    tot_inter = arr_intermediate.sum()
    y_inter, x_inter = np.indices(arr_intermediate.shape)
    cx_inter = float(np.sum(x_inter * arr_intermediate) / tot_inter)
    cy_inter = float(np.sum(y_inter * arr_intermediate) / tot_inter)

    # Fine residual translation needed to hit 13.50
    sub_dx = 13.5 - cx_inter
    sub_dy = 13.5 - cy_inter

    # Ensure translation does not push stroke pixels out of the 28x28 bounds
    inter_rows = np.where(arr_intermediate > 10.0)[0]
    inter_cols = np.where(arr_intermediate > 10.0)[1]
    curr_min_x, curr_max_x = int(inter_cols.min()), int(inter_cols.max())
    curr_min_y, curr_max_y = int(inter_rows.min()), int(inter_rows.max())

    # Clamp residual shift to maintain inside-canvas guarantee
    sub_dx = max(-curr_min_x, min(27 - curr_max_x, sub_dx))
    sub_dy = max(-curr_min_y, min(27 - curr_max_y, sub_dy))

    if abs(sub_dx) > 0.01 or abs(sub_dy) > 0.01:
        final_img = canvas_intermediate.transform(
            (28, 28),
            Image.Transform.AFFINE,
            (1, 0, -sub_dx, 0, 1, -sub_dy),
            resample=Image.Resampling.BILINEAR,
        )
    else:
        final_img = canvas_intermediate

    final_b64 = image_to_base64(final_img)

    # Calculate actual final center of mass and final bounding box on the 28x28 tensor
    final_arr = np.array(final_img, dtype=np.float32)
    tot_final = final_arr.sum()
    y_fin, x_fin = np.indices(final_arr.shape)
    final_cx = float(np.sum(x_fin * final_arr) / tot_final)
    final_cy = float(np.sum(y_fin * final_arr) / tot_final)

    fin_rows = np.where(final_arr > 10.0)[0]
    fin_cols = np.where(final_arr > 10.0)[1]
    if len(fin_rows) > 0 and len(fin_cols) > 0:
        final_bbox = [int(fin_cols.min()), int(fin_rows.min()), int(fin_cols.max()), int(fin_rows.max())]
    else:
        final_bbox = [0, 0, 27, 27]

    total_translation_x = round(paste_x + sub_dx, 2)
    total_translation_y = round(paste_y + sub_dy, 2)

    # 6. Pixel Normalization to [0.0, 1.0]
    normalized_arr = np.clip(final_arr / 255.0, 0.0, 1.0)

    # 7. Convert to PyTorch Tensor: shape (1, 1, 28, 28)
    tensor = torch.from_numpy(normalized_arr).float().unsqueeze(0).unsqueeze(0)

    # Pixel statistics
    min_val = float(normalized_arr.min())
    max_val = float(normalized_arr.max())
    mean_val = float(normalized_arr.mean())
    active_count = int(np.count_nonzero(normalized_arr > 0.05))

    debug_info = {
        "original_image": original_b64,
        "cropped_image": cropped_b64,
        "cropped_dimensions": [box_w, box_h],
        "resized_image": resized_b64,
        "resized_dimensions": [new_w, new_h],
        "final_image": final_b64,
        "final_dimensions": [28, 28],
        # Explicit Center of Mass and Translation Breakdown
        "original_center_of_mass": [round(orig_cx, 2), round(orig_cy, 2)],
        "final_center_of_mass": [round(final_cx, 2), round(final_cy, 2)],
        "horizontal_translation": total_translation_x,
        "vertical_translation": total_translation_y,
        "translation_applied": [total_translation_x, total_translation_y],
        "final_bounding_box": final_bbox,
        # Normalization and Tensor Stats
        "min_value": round(min_val, 4),
        "max_value": round(max_val, 4),
        "mean_value": round(mean_val, 4),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "normalization_confirmed": bool(0.0 <= min_val and max_val <= 1.0),
        "active_pixels": active_count,
        "pixel_sample": np.round(normalized_arr, 2).tolist(),
    }

    return tensor, final_b64, debug_info


def predict(model: DigitMLP, tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Runs model inference with torch.no_grad().
    Returns predicted digit, confidence, and all 10 probabilities.
    """
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        predicted_digit = int(torch.argmax(probs).item())
        confidence = float(probs[predicted_digit].item())
        probabilities = [round(float(p.item()), 4) for p in probs]

    return {
        "predicted_digit": predicted_digit,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
    }
