# 📘 API Reference — AutoML Model Comparator

Full technical documentation for the `automl_comparator` package.

---

## 1. Module Overview

| Module | Purpose |
| --- | --- |
| `api.py` | Public `compare_models()` entry point |
| `comparator.py` | Core engine + metric definitions + `run_experiments()` |
| `detector.py` | Problem-type validation (`is_classification`) |
| `leaderboard.py` | Build the sorted leaderboard DataFrame |
| `models.py` | Model catalogues, `resolve_models()`, `get_model_names()` |
| `scalers.py` | Scaler catalogue, pipeline construction |

---

## 2. Main Function

### `compare_models(...)`

The single entry point users call most of the time.

**Signature**

```python
compare_models(
    X_train, y_train, X_test, y_test,
    *,
    is_classification,       # required
    scale=False,             # opt-in scaling
    scale_columns=None,
    models=None,
    scalers=None,
    metrics=None,
    problem=None,
    sort_by=None,
    ascending=False,
    n_jobs=1,
) -> pandas.DataFrame
```

**Parameters**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `X_train` | array-like (2D) | — | Training features |
| `y_train` | array-like (1D) | — | Training target |
| `X_test` | array-like (2D) | — | Test features |
| `y_test` | array-like (1D) | — | Test target |
| `is_classification` | `bool` | **required** | `True` → classification, `False` → regression. Must be passed explicitly. |
| `scale` | `bool`/`str`/`list[str]` | `False` | Scaling select. `False` → **raw** data only. `True` → every technique. `"StandardScaler"` → that one. `["MinMaxScaler","RobustScaler"]` → those only. Unknown name → `ValueError`. |
| `scale_columns` | `list[int]`/`None` | `None` | Column indices to scale (active when scaling is enabled). `None` → **all** columns scaled; `[1,2,4]` → only those columns. |
| `models` | `str`/`list`/`type`/`dict`/`None` | `None` | Which models to run. `None` → all built-in. |
| `scalers` | `list[str]`/`None` | `None` | Exact scaler names to try. Overrides `scale` when provided. `None` → decided by `scale`. |
| `metrics` | `dict`/`None` | `None` | Custom `{name: callable}` metrics. `None` → built-in set. |
| `problem` | `"classification"`/`"regression"`/`None` | `None` | Optional cross-check only. If provided, must match `is_classification`. |
| `sort_by` | `str`/`None` | `None` | Metric to sort leaderboard by. `None` → default (Accuracy/R²). |
| `ascending` | `bool` | `False` | `False` → best on top; `True` → worst on top. |
| `n_jobs` | `int` | `1` | Parallel jobs. `1` → sequential. |

> **Note:** `is_classification` is now **required** — there is no automatic
> detection. It raises `TypeError` if omitted and `ValueError` if it conflicts
> with an explicit `problem`.

> **Scaling is opt-in.** Without `scale=True` the library runs only on the raw,
> unscaled data. Pass `scale=True` (optionally with `scale_columns`) to enable
> the full set of scaling techniques.

**Returns**

A `pandas.DataFrame` (the leaderboard). Columns: `Model`, `Scaling`,
`Scaled Columns`, plus one column per metric.

---

## 3. Underlying Functions

### `run_experiments(...)` *(in `comparator.py`)*

The lower-level engine that actually runs every (model × scaler) combination
and returns a list of result dicts — **without** building/printing the
DataFrame. `compare_models()` wraps this.

```python
results = run_experiments(
    X_train, y_train, X_test, y_test,
    is_classification=True,   # required: True=classification, False=regression
    scale=True,               # opt-in scaling
    scale_columns=[0, 2],     # scale only columns 0 and 2 (None = all)
)
# results: [ {"model_name": ..., "scaler_name": ...,
#             "scaled_columns_str": ..., "metrics": {...}, "error": None}, ... ]
```

### `build_leaderboard(results, sort_by=None, ascending=False)` *(in `leaderboard.py`)*

Converts the list of result dicts into a sorted `pandas.DataFrame`.

### `is_classification(y)` *(in `detector.py`)*

Heuristic helper used to **validate** the required `is_classification` flag
against your data. `compare_models` / `run_experiments` do **not** auto-detect — if the flag you pass does not match `y_train`, they raise a `ValueError`.
You can call this helper yourself to check your target and pick the right flag.
Returns `True` if `y` is:
- object / string / bool / category dtype, **or**
- numeric with ≤ 20 unique **integer** values.

### `resolve_models(models, problem)` *(in `models.py`)*

Normalises any valid `models` input into a `{name: estimator_class}` dict.

### `get_model_names(problem)` *(in `models.py`)*

Returns the list of available model names for a problem type.

---

## 4. Model Catalogues

### Classification (`CLASSIFICATION_MODELS`)

**Linear:** Logistic Regression, SGD Classifier, Perceptron,
Passive Aggressive Classifier

**Tree-based:** Decision Tree, Random Forest, Extra Trees, Gradient Boosting,
Hist Gradient Boosting

**Distance-based:** KNN

**Support vector:** SVC, Linear SVC, Nu SVC

**Naive Bayes:** Gaussian NB, Multinomial NB, Bernoulli NB, Complement NB

**Discriminant analysis:** LDA, QDA

**Ensemble:** AdaBoost, **XGBoost** (optional)

### Regression (`REGRESSION_MODELS`)

**Linear:** Linear Regression, Ridge, Lasso, ElasticNet, Bayesian Ridge,
Huber, Quantile, SGD Regressor, Passive Aggressive Regressor

**Tree-based:** Decision Tree, Random Forest, Extra Trees, Gradient Boosting,
Hist Gradient Boosting

**Distance-based:** KNN, Radius Neighbors

**Support vector:** SVR, Linear SVR, Nu SVR

**Ensemble:** AdaBoost Regressor, **XGBoost** (optional)

---

## 5. Scaling Techniques (`SCALERS`)

| Name | sklearn class |
| --- | --- |
| `StandardScaler` | Z-score normalisation |
| `MinMaxScaler` | Scale to [0, 1] |
| `RobustScaler` | Robust to outliers (uses median/IQR) |
| `MaxAbsScaler` | Scale by max absolute value |
| `Normalizer` | Scale rows to unit norm |
| `PowerTransformer` | Power / Box-Cox / Yeo-Johnson |
| `QuantileTransformer` | Non-linear, maps to uniform/Gaussian |

Plus the special **`No Scaling`** option (default first).

---

## 6. Default Metrics

### Classification

| Metric | Direction |
| --- | --- |
| Accuracy | higher better |
| Precision | higher better |
| Recall | higher better |
| F1 | higher better |

### Regression

| Metric | Direction |
| --- | --- |
| R² | higher better |
| MAE | lower better |
| MSE | lower better |
| RMSE | lower better |

**Important:** sorting is descending by default. For regression metrics where
*lower is better* (MAE/MSE/RMSE), pass `ascending=True`.

---

## 7. Custom Metrics

Pass your own metric functions. Each function must accept
`(y_true, y_pred)` and return a scalar.

```python
from sklearn.metrics import log_loss

def my_metric(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

compare_models(
    X_train, y_train, X_test, y_test,
    is_classification=True,
    metrics={"CustomMetric": my_metric},
    sort_by="CustomMetric",
)
```

---

## 8. Example: Full Workflow (Regression)

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from automl_comparator import compare_models

X, y = make_regression(n_samples=300, n_features=5, noise=10, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

lb = compare_models(
    Xtr, ytr, Xte, yte,
    is_classification=False,
    scale=True,
    scale_columns=[0, 1, 3],
    models=["XGBoost", "Random Forest", "Ridge"],
    sort_by="R2",
)
```

---

## 9. Error Handling

- **Unknown model name** → raises `ValueError` listing available models.
- **Unknown scaler name** → raises `ValueError` listing available scalers.
- **Unknown `sort_by` metric** → raises `ValueError` listing available metrics.
- **Model/scaler incompatibility** (e.g. MultinomialNB with negative values) →
  that single experiment is skipped gracefully; the `error` field records it and
  the rest continue.

**Caution:** `sort_by` values are matched *exactly* (e.g. `"F1"`, `"R2"`,
`"Accuracy"`).
