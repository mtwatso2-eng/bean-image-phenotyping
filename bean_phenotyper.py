"""Backward-compatible entry point for the bean phenotyping pipeline."""

from onnx_sam3 import get_model, phenotype_image, warm_predictor

__all__ = ["get_model", "phenotype_image", "warm_predictor"]
