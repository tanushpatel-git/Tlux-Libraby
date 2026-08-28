# ❓ Questions & Answers — AutoML Model Comparator

Common questions (and answers) about the library — the **what**, **how**, and
**why** behind every feature, including **pros & cons**.

---

## 🔍 General

### 1. What does this library actually do?

It automatically trains **multiple ML models** × **multiple scaling techniques**,
computes metrics on a held-out test set, and returns a **leaderboard** showing
which combination performed best.

### 2. Do I need to split my data myself?

Yes. The library expects you to pass `X_train, y_train, X_test, y_test`
already split. This keeps the library simple and puts the train/test split
under your control. (Cross-validation is a planned future feature.)

### 3. Classification or regression — do I pick?

**Yes — you must.** The library does **not** auto-detect. Pass the required
`is_classification=True` (classification) or `is_classification=False`
(regression) flag:

```python
compare_models(..., is_classification=True)    # classification
compare_models(..., is_classification=False)   # regression
```

If you aren't sure, you can use the `is_classification(y)` helper to check your
target, then hand its result to the call:

```python
from tlux import is_classification
compare_models(..., is_classification=is_classification(y_train))
```

**The flag is validated against your data.** If the flag you pass does not
match `y_train`, the call **raises a `ValueError`** instead of running — no
silent auto-detection, no guessing:

- `is_classification=True` on regression/continuous `y` → error, telling you to use `False`.
- `is_classification=False` on categorical / few-unique-integer `y` → error, telling you to use `True`.

| `is_classification(y)` returns `True` when | |
| --- | --- |
| `y` is strings / booleans / categories | Classification |
| `y` is numeric with ≤ 20 unique *integer* values | Classification |
| Otherwise (continuous float) | Regression |

---

## ⚙️ Scaling

### 4. When does scaling happen, and how do I control it?

Scaling is **opt-in**. By default the library runs on the **raw, unscaled**
data only. The `scale` argument lets you pick **which** scaling techniques run:

```python
compare_models(..., scale=False)                             # raw data only (default)
compare_models(..., scale=True)                              # try EVERY technique
compare_models(..., scale="StandardScaler")                  # just this one
compare_models(..., scale=["MinMaxScaler", "RobustScaler"])  # just these
```

An unknown scaler name raises a `ValueError` listing the available options.

To scale only **specific** columns, add `scale_columns` (active when scaling is
enabled):

```python
compare_models(..., scale="RobustScaler", scale_columns=[0, 2, 4])
```

Columns **not** in `scale_columns` are left **unchanged** (passed through as-is).

### 5. Which scaling techniques are available?

`No Scaling, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
Normalizer, PowerTransformer, QuantileTransformer`. You can also bypass
`scale` entirely and pass the exact list via `scalers=[...]` (highest
priority).

### 6. Why would I scale only some columns?

- **Scale-sensitive models** (SVM, KNN, logistic regression, neural nets)
  need features on a similar scale.
- **Scale-insensitive models** (trees, random forests, boosting) don't care, so
  scaling them wastes time and can confuse interpretation.
- Letting the library try *both* shows you whether scaling actually helps.

| Pros of scaling | Cons of scaling |
| --- | --- |
| Better for distance/linear models | Unnecessary for tree models |
| Speeds up gradient-based training | Can hide original units/meaning |
| Handles outliers (Robust/Power) | Some scalers fail on negative values |

---

## 📊 Metrics & Leaderboard

### 7. Which metrics are computed?

- **Classification:** Accuracy, Precision, Recall, F1-score
- **Regression:** R², MAE, MSE, RMSE

### 8. How do I sort by a specific metric (e.g. F1)?

```python
compare_models(..., sort_by="F1")
```

**Defaults:** classification → `Accuracy`; regression → `R2`.

### 9. What should I sort by, Accuracy or F1?

- **Accuracy** = just % correct. Fine for **balanced** classes.
- **F1-score** = harmonic mean of precision & recall. **Better for imbalanced
  data** (a model that predicts the majority class every time gets high accuracy
  but low F1).

### 10. For regression, lower is better for some metrics — how do I sort?

```python
compare_models(..., sort_by="MAE", ascending=True)   # smallest error on top
```

Sorting is **descending by default**; pass `ascending=True` for metrics where
lower is better (MAE/MSE/RMSE).

---

## 🤖 Models

### 11. How do I run ONLY the models I want?

```python
compare_models(..., models="XGBoost")                    # one by name
compare_models(..., models=["XGBoost", "KNN"])           # list of names
from sklearn.svm import SVC
compare_models(..., models=[SVC])                        # sklearn classes
compare_models(..., models={"My Model": SVC})            # custom dict
```

### 12. How do I see what models are available?

```python
from automl_comparator import get_model_names
print(get_model_names("classification"))
print(get_model_names("regression"))
```

### 13. Why isn't XGBoost showing up?

XGBoost is an **optional** dependency. Install it:

```bash
pip install xgboost
```

If it's not installed, the library silently skips it.

### 14. Can I add my own estimator?

Yes — pass its class (or a `{name: class}` dict). It must be sklearn-compatible
(implement `fit` and `predict`).

---

## ⚡ Performance

### 15. My dataset produces hundreds of experiments — it's slow. Help!

- Limit the models: `models=["XGBoost", "Random Forest"]`
- Limit the scalers: `scalers=["StandardScaler"]`
- Use parallel training: `n_jobs=-1` (uses all CPU cores)

```python
compare_models(..., models=["KNN", "SVC"], scalers=["StandardScaler"], n_jobs=-1)
```

### 16. What if some (model, scaler) combinations fail?

They're skipped gracefully. Each result records an `error` field; the rest of
the experiments keep running. Example failure: MultinomialNB cannot accept
negative values.

---

## 🧩 Design

### 17. What are the main Pros & Cons of the library as a whole?

| Pros | Cons |
| --- | --- |
| Saves a lot of repetitive ML code | You must split data yourself (no CV yet) |
| Tests many models & scalers at once | No hyperparameter tuning yet |
| Column-level scaling control | No categorical encoding / missing-value handling |
| Handles classification & regression | Some defaults (e.g. RBF SVR) may under-fit |
| Clean leaderboard DataFrame | Uses default hyperparameters only |
| Comparative & explainable | Training all combos can be slow (use `n_jobs`) |

### 18. What are the built-in models vs external ones?

- **Built-in:** all `scikit-learn` estimators (no extra install).
- **Optional/external:** XGBoost (`pip install xgboost`).

### 19. Why is `No Scaling` tried by default?

Because scaling isn't always beneficial (e.g. tree models). Including it gives
you a fair baseline and shows the real impact of scaling.

---

## 🔮 Future

### 20. What's planned next?

Automatic preprocessing, missing-value handling, categorical encoding,
cross-validation, hyperparameter tuning, feature selection/importance,
confusion matrix, ROC-AUC, Precision-Recall, model explainability,
best-model recommendation, experiment reports, and model save/load.

---

## 🆘 Troubleshooting

### 21. "Unknown model '...'"

The name you passed isn't in the catalogue (likely a typo or XGBoost not
installed). Check with `get_model_names(...)`.

### 22. "sort_by='...' not found"

The metric name must match exactly (e.g. `"F1"`, `"R2"`, `"Accuracy"`). Check
the error message for the available list.

### 23. All my metrics are `NaN`

That experiment failed (recorded in `error`, and printed to stderr). Inspect
the result dicts' `error` field to see why.
