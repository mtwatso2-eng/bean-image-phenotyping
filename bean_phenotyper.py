import copy
import os
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np

from model_cache import get_model_path

SEGMENTATION_PROMPT = "a bean"

_predictor = None


def _check_runtime():
    import torch

    if not hasattr(torch.nn, "attention"):
        raise RuntimeError(
            "SAM3 requires PyTorch 2.3 or newer with torch.nn.attention. "
            f"Installed torch {torch.__version__}. "
            "Recreate your virtual environment with Python 3.10+ and run "
            "pip install -r requirements.txt."
        )


def get_predictor():
    """Load the SAM3 predictor lazily on first use."""
    global _predictor
    if _predictor is None:
        _check_runtime()
        from ultralytics.models.sam import SAM3SemanticPredictor

        overrides = dict(
            conf=0.4,
            task="segment",
            mode="predict",
            model=str(get_model_path()),
            quantize="fp16",
            save=False,
        )
        _predictor = SAM3SemanticPredictor(overrides=overrides)
    return _predictor


def warm_predictor():
    """Download weights if needed and load the predictor into memory."""
    _check_runtime()
    predictor = get_predictor()
    # Force a full model load so startup fails fast if inference cannot run.
    if hasattr(predictor, "setup_model"):
        predictor.setup_model()


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
    predictor = get_predictor()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img_bgr)

    try:
        predictor.set_image(tmp_path)
        results = predictor(text=[SEGMENTATION_PROMPT])
    finally:
        os.unlink(tmp_path)

    if not results:
        overlay_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return overlay_rgb, []

    first_result = results[0]
    orig_img = first_result.orig_img
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    overlay_image = copy.deepcopy(orig_img).astype(np.float32)
    num_colors_to_use = 10
    colormap = plt.colormaps["tab10"]
    colors_rgb_float = [colormap(i)[:3] for i in range(num_colors_to_use)]
    colors_bgr_uint8 = [(np.array(c) * 255).astype(np.uint8)[::-1] for c in colors_rgb_float]

    alpha = 0.5
    color_idx = 0
    bean_records = []

    for result in results:
        if result.masks is None:
            continue

        boxes = result.boxes.xywh.cpu().numpy()

        for i in range(result.masks.data.shape[0]):
            mask_np = result.masks.data[i].cpu().numpy().astype(bool)

            _, _, box_w, box_h = boxes[i]
            ratio = box_h / box_w if box_w != 0 else 0

            masked_pixels = orig_img_rgb[mask_np]
            if masked_pixels.size > 0:
                mean_rgb = np.round(masked_pixels.mean(axis=0)).astype(int)
                color_variance = np.mean(np.std(masked_pixels, axis=0))
            else:
                mean_rgb = [0, 0, 0]
                color_variance = 0

            bean_records.append(
                {
                    "image": image_name,
                    "bean_id": i,
                    "width": box_w,
                    "height": box_h,
                    "ratio_h_w": ratio,
                    "mean_r": mean_rgb[0],
                    "mean_g": mean_rgb[1],
                    "mean_b": mean_rgb[2],
                    "color_variance": color_variance,
                }
            )

            current_color_bgr = colors_bgr_uint8[color_idx % len(colors_bgr_uint8)]
            color_idx += 1
            color_float = np.array(current_color_bgr).astype(np.float32)
            overlay_image[mask_np] = (1 - alpha) * overlay_image[mask_np] + alpha * color_float

    overlay_rgb = overlay_image.astype(np.uint8)[:, :, ::-1]
    return overlay_rgb, bean_records
