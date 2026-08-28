"""tlux — automatically compare ML models with different scalers."""

from .api import compare_models
from .comparator import run_experiments
from .detector import is_classification
from .leaderboard import build_leaderboard
from .models import CLASSIFICATION_MODELS, REGRESSION_MODELS, get_model_names
from .scalers import SCALERS, get_scaler_names

__all__ = [
    "compare_models",
    "run_experiments",
    "is_classification",
    "build_leaderboard",
    "CLASSIFICATION_MODELS",
    "REGRESSION_MODELS",
    "get_model_names",
    "SCALERS",
    "get_scaler_names",
]
