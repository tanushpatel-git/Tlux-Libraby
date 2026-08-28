# tlux_lazy

An **open-source Python library** that automatically compares different Machine
Learning models along with different preprocessing and scaling techniques.

Instead of manually writing *"model → choose scaler → choose columns →
transform → train → predict → calculate metrics"* for every combination, this
library runs all the experiments for you and returns a **model leaderboard**.

---

## Features

- **Auto problem-type detection** — automatically figures out if your data is a
  classification or regression problem.
- **21 classifiers + 21 regressors** — including XGBoost (optional).
- **8 scaling techniques** — Standard, MinMax, Robust, MaxAbs, Normalizer,
  Power, Quantile, plus *No Scaling*.
- **User-controlled column scaling** — pick *exactly* which columns get scaled
  via `scale_columns`.
- **Automated leaderboard** — sorted by the metric you choose (default:
  Accuracy for classification, R² for regression).
- **7 default metrics** — Accuracy / Precision / Recall / F1 (classification),
  R² / MAE / MSE / RMSE (regression). Fully customisable.
- **Run only the models you want** — pass a name, list, class, or dict.
- **Parallel training** via `n_jobs` (joblib).
- **Rich output** — leaderboard table + detailed per-experiment breakdown.

---

## Installation

```bash
pip install tlux_lazy
```

For development (includes tests):

```bash
pip install -e ".[dev]"
```

For XGBoost support (optional):

```bash
pip install -e ".[xgboost]"
```

Or install everything at once:

```bash
pip install -e ".[dev,xgboost]"
```

---

## Quick Start

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tlux import compare_models

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Compare all models, scaling only columns 0 and 2
leaderboard = compare_models(
    X_train, y_train, X_test, y_test,
    scale_columns=[0, 2],
)
```

Output is a `pandas.DataFrame` (the leaderboard) and a printed table:

| Model | Scaling | Scaled Columns | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| SVC | StandardScaler | 0, 2 | 1.00 | 1.00 | 1.00 | 1.00 |
| XGBoost | No Scaling | 0, 2 | 1.00 | 1.00 | 1.00 | 1.00 |
| Random Forest | RobustScaler | 0, 2 | 1.00 | 1.00 | 1.00 | 1.00 |
| ... | ... | ... | ... | ... | ... | ... |

---

## Run Only Specific Models

```python
# Single model by name
compare_models(..., models="XGBoost")

# A few models by name
compare_models(..., models=["XGBoost", "Random Forest", "KNN"])

# Using sklearn estimator classes directly
from sklearn.svm import SVC
compare_models(..., models=[SVC, RandomForestClassifier])

# Custom-named dict (great for custom estimators)
compare_models(..., models={"My Custom Model": SVC})
```

List available model names first:

```python
from tlux import get_model_names

print(get_model_names("classification"))
print(get_model_names("regression"))
```

---

## Sort by Any Metric

```python
# Sort the leaderboard by F1-score
compare_models(..., sort_by="F1")

# Sort by MAE (ascending — lower is better)
compare_models(..., sort_by="MAE", ascending=True)

# Regression defaults to R2, classification defaults to Accuracy
```

---

## Detailed Experiment Output

```python
compare_models(..., detailed=True)
```

Prints a full breakdown for every experiment:

```text
--- Experiment: SVC ---
  Preprocessing : StandardScaler
  Scaled Columns: 0, 2
  Accuracy       : 1.0
  Precision      : 1.0
  Recall         : 1.0
  F1             : 1.0
```

---

## Workflow

```
X_train, y_train, X_test, y_test
              ↓
       Detect Problem Type
          ↙          ↘
 Classification     Regression
       ↓                ↓
 Try Classifiers    Try Regressors
       ↓                ↓
 Try Preprocessing / Scaling
       ↓
 Train & Predict
       ↓
 Calculate Metrics
       ↓
 Compare Results
       ↓
      Leaderboard
```

---

## Project Structure

```
Tlux/
├── pyproject.toml
├── README.md
├── DOCUMENTATION.md
├── WHY.md
├── Q&A.md
├── tlux/
│   ├── __init__.py      # Public exports
│   ├── api.py           # compare_models() entry point
│   ├── comparator.py    # Core engine: runs experiments
│   ├── detector.py      # Auto-detect classification vs regression
│   ├── leaderboard.py   # Build + print leaderboard
│   ├── models.py        # Model catalogues + model resolution
│   └── scalers.py       # Scaling techniques + ColumnTransformer
└── tests/
    └── test_tlux.py     # Test suite (pytest)
```

---

## Public API

```python
from tlux import (
    compare_models,       # Main entry point
    run_experiments,      # Run experiments manually
    is_classification,    # Detect problem type
    build_leaderboard,    # Build leaderboard from results
    print_leaderboard,    # Pretty-print leaderboard
    CLASSIFICATION_MODELS, # List of classification model names
    REGRESSION_MODELS,     # List of regression model names
    get_model_names,       # Get model names by problem type
    SCALERS,               # List of scaler names
    get_scaler_names,      # Get available scaler names
)
```

---

## Future Features

- Automatic preprocessing
- Missing-value handling
- Categorical encoding
- Cross-validation
- Hyperparameter tuning
- Feature selection / importance
- Confusion matrix, ROC-AUC, Precision-Recall
- Model explainability
- Best-model recommendation
- Experiment reports
- Model saving / loading
- Custom scalers & custom metrics

---

## License

MIT License. Free for the Python and Machine Learning community.

---

See [DOCUMENTATION.md](./DOCUMENTATION.md) for the full API reference,
[Q&A.md](./Q&A.md) for common questions, and [WHY.md](./WHY.md) for the
design rationale.
