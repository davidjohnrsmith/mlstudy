"""Deployable inference artifacts and model reconstruction.

This module provides:
- Portable model export formats (JSON, numpy arrays)
- Pure numpy inference for linear models
- XGBoost Booster save/load via JSON/UBJ
- LightGBM Booster save/load via model text files
"""

from __future__ import annotations

# from mlstudy.deploy.inference.linear import LinearInferenceModel
#
# __all__ = [
#     "InferenceModel",
#     "LinearInferenceModel",
#     "export_model",
#     "get_model_type",
#     "load_inference_model",
# ]
