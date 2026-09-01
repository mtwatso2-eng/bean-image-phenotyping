import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).parent
SAM3_ONNX_REPO_ID = "rusen/sam3-browser-int8"

# Avoid CoreML on macOS for these models; it is slow and partially unsupported.
os.environ.setdefault("SAMEXPORTER_ONNX_PROVIDERS", "cpu")

SAM3_VARIANTS = {
    # Lower-resolution ONNX models for memory-constrained hosts (~4 GB RAM).
    "compact-448": {
        "image_encoder": "compact-448/sam3_image_encoder_fp16.onnx",
        "language_encoder": "compact-448/sam3_language_encoder.onnx",
        "decoder": "compact-448/sam3_decoder.onnx",
    },
    "int8": {
        "image_encoder": "sam3_image_encoder.onnx",
        "language_encoder": "sam3_language_encoder.onnx",
        "decoder": "sam3_decoder.onnx",
    },
}

load_dotenv(APP_DIR / "tokens.env")

_paths: "Sam3OnnxPaths | None" = None


@dataclass(frozen=True)
class Sam3OnnxPaths:
    image_encoder: Path
    language_encoder: Path
    decoder: Path


def get_sam3_variant() -> str:
    variant = os.environ.get("SAM3_ONNX_VARIANT", "compact-448").strip()
    if variant not in SAM3_VARIANTS:
        raise ValueError(
            f"Unknown SAM3_ONNX_VARIANT '{variant}'. "
            f"Choose one of: {', '.join(SAM3_VARIANTS)}"
        )
    return variant


def _download_model(filename: str) -> Path:
    local_override = os.environ.get("SAM3_ONNX_DIR")
    if local_override:
        path = Path(local_override).expanduser() / filename
        if path.exists():
            return path
        raise FileNotFoundError(f"SAM3 ONNX file not found: {path}")

    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    downloaded_path = hf_hub_download(
        repo_id=SAM3_ONNX_REPO_ID,
        filename=filename,
        token=token,
    )
    return Path(downloaded_path)


def get_model_paths() -> Sam3OnnxPaths:
    """Return cached paths to the SAM3 ONNX model files."""
    global _paths
    if _paths is not None:
        return _paths

    variant = get_sam3_variant()
    files = SAM3_VARIANTS[variant]
    _paths = Sam3OnnxPaths(
        image_encoder=_download_model(files["image_encoder"]),
        language_encoder=_download_model(files["language_encoder"]),
        decoder=_download_model(files["decoder"]),
    )
    return _paths
