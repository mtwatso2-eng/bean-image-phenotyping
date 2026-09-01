from dataclasses import dataclass, field
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sam3_runtime.sam3_onnx import (
    SAM3ImageDecoder,
    SAM3ImageEncoder,
    SAM3LanguageEncoder,
    SegmentAnything3ONNX,
)

from model_cache import Sam3OnnxPaths, get_model_paths

SEGMENTATION_PROMPT = "a bean"
TEXT_ONLY_PROMPT = [{"type": "text", "data": SEGMENTATION_PROMPT}]

_model = None


@dataclass
class Sam3OnnxRunner:
    """Run SAM3 ONNX models with cached sessions for repeated inference."""

    paths: Sam3OnnxPaths
    _image_encoder: SAM3ImageEncoder | None = field(default=None, init=False, repr=False)
    _language_encoder: SAM3LanguageEncoder | None = field(default=None, init=False, repr=False)
    _decoder: SAM3ImageDecoder | None = field(default=None, init=False, repr=False)

    @property
    def image_encoder(self) -> SAM3ImageEncoder:
        if self._image_encoder is None:
            self._image_encoder = SAM3ImageEncoder(self.paths.image_encoder)
        return self._image_encoder

    @property
    def language_encoder(self) -> SAM3LanguageEncoder:
        if self._language_encoder is None:
            self._language_encoder = SAM3LanguageEncoder(self.paths.language_encoder)
        return self._language_encoder

    @property
    def decoder(self) -> SAM3ImageDecoder:
        if self._decoder is None:
            self._decoder = SAM3ImageDecoder(self.paths.decoder)
        return self._decoder

    def encode(self, cv_image: np.ndarray, text_prompt: str | None = None) -> dict[str, Any]:
        if cv_image is None or cv_image.ndim != 3 or cv_image.shape[2] != 3:
            raise ValueError("Expected a non-empty BGR image with three channels")

        original_size = cv_image.shape[:2]
        image_encoder_outputs = self.image_encoder(cv_image)

        text_prompt = text_prompt or "visual"
        lang_outputs = self.language_encoder(text_prompt)

        return {
            "vision_pos_enc_0": image_encoder_outputs[0],
            "vision_pos_enc_1": image_encoder_outputs[1],
            "vision_pos_enc_2": image_encoder_outputs[2],
            "backbone_fpn_0": image_encoder_outputs[3],
            "backbone_fpn_1": image_encoder_outputs[4],
            "backbone_fpn_2": image_encoder_outputs[5],
            "original_size": original_size,
            "language_mask": lang_outputs[0],
            "language_features": lang_outputs[1],
            "language_embeds": lang_outputs[2],
        }

    def predict_masks(
        self,
        embedding: dict[str, Any],
        prompt,
        confidence_threshold: float = 0.4,
        nms_threshold: float | None = 0.7,
        nms_mode: str = "mask",
        max_instances: int | None = None,
    ) -> np.ndarray:
        helper = SegmentAnything3ONNX.__new__(SegmentAnything3ONNX)
        helper.decoder = self.decoder
        return helper.predict_masks(
            embedding,
            prompt,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            nms_mode=nms_mode,
            max_instances=max_instances,
            prefer_prompted_region=False,
        )


def get_model() -> Sam3OnnxRunner:
    global _model
    if _model is None:
        _model = Sam3OnnxRunner(get_model_paths())
    return _model


def warm_predictor():
    """Download ONNX weights if needed and load all model sessions."""
    model = get_model()
    _ = model.decoder
    _ = model.language_encoder
    _ = model.image_encoder


def _mask_bbox(mask_np: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask_np)
    if len(xs) == 0:
        return 0.0, 0.0
    return float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)


def phenotype_image(img_bgr, image_name):
    """
    Segment beans in an image and compute phenotypic measurements.

    Args:
        img_bgr: Input image as a numpy array in BGR format
        image_name: Filename used when recording results

    Returns:
        tuple: (overlay_rgb, bean_records) where overlay_rgb is a visualization
               and bean_records is a list of measurement dicts
    """
    model = get_model()
    embedding = model.encode(img_bgr, text_prompt=SEGMENTATION_PROMPT)
    masks = model.predict_masks(
        embedding,
        TEXT_ONLY_PROMPT,
        confidence_threshold=0.4,
    )

    if masks is None or len(masks) == 0:
        overlay_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return overlay_rgb, []

    orig_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay_image = img_bgr.astype(np.float32).copy()
    num_colors_to_use = 10
    colormap = plt.colormaps["tab10"]
    colors_rgb_float = [colormap(i)[:3] for i in range(num_colors_to_use)]
    colors_bgr_uint8 = [(np.array(c) * 255).astype(np.uint8)[::-1] for c in colors_rgb_float]

    alpha = 0.5
    bean_records = []

    for bean_id, instance_mask in enumerate(masks[:, 0]):
        mask_np = instance_mask.astype(bool)
        box_w, box_h = _mask_bbox(mask_np)
        if box_w == 0 or box_h == 0:
            continue

        ratio = box_h / box_w if box_w != 0 else 0
        masked_pixels = orig_img_rgb[mask_np]
        if masked_pixels.size > 0:
            mean_rgb = np.round(masked_pixels.mean(axis=0)).astype(int)
            color_variance = float(np.mean(np.std(masked_pixels, axis=0)))
        else:
            mean_rgb = [0, 0, 0]
            color_variance = 0.0

        bean_records.append(
            {
                "image": image_name,
                "bean_id": bean_id,
                "width": box_w,
                "height": box_h,
                "ratio_h_w": ratio,
                "mean_r": int(mean_rgb[0]),
                "mean_g": int(mean_rgb[1]),
                "mean_b": int(mean_rgb[2]),
                "color_variance": color_variance,
            }
        )

        current_color_bgr = colors_bgr_uint8[bean_id % len(colors_bgr_uint8)]
        color_float = np.array(current_color_bgr).astype(np.float32)
        overlay_image[mask_np] = (1 - alpha) * overlay_image[mask_np] + alpha * color_float

    overlay_rgb = overlay_image.astype(np.uint8)[:, :, ::-1]
    return overlay_rgb, bean_records
