# 📄 Why We Built This — AutoML Model Comparator

This file explains the **motivation**, **problem**, **why each function
exists**, and **what decision each API choice represents**. It's the "design
story" behind the library.

---

## 1. The Problem We Were Solving

When training a Machine Learning model, data scientists repeat the same
tedious, manual loop over and over:

```
Take a model
    → decide a scaler
    → decide which columns to scale
    → transform the data
    → train
    → predict
    → compute metrics
    → repeat for the NEXT model/scaler
```

For a single dataset with, say, 20 models and 8 scalers, that's **160 manual
experiments** — mostly copy-paste code. This is:

- **Time-consuming**
- **Error-prone** (easy to forget a scaler or leak test data)
- **Hard to compare fairly**

**The goal:** let the user provide data once and get back a clean
**leaderboard** comparing every (model × scaling) combination automatically.

---

## 2. The Core Design Decision: Scaling Is Opt-In + User Controls Columns

A key idea in this project is that **the user should stay in control of
scaling**, not the library. By default the library runs on the **raw, unscaled
data** — no hidden preprocessing. The `scale` argument opts in and picks
**which** technique(s) run, and `scale_columns` chooses exactly which columns
to scale:

```python
scale=False                          # raw data only (default)
scale=True                           # every scaling technique
scale="StandardScaler"               # just one technique
scale=["MinMaxScaler", "RobustScaler"]  # just a few
scale="RobustScaler", scale_columns=[0, 2, 4]  # limit columns too
```

**Why this matters:**

- Not every column needs scaling (e.g. a binary flag, a small-count integer).
- Different domains care about different features, and different techniques
  suit different data (outliers → Robust, log-like → Power).
- Automatic "scale everything by default" can **hurt** — e.g. scaling a sparse
  or boolean feature is often meaningless, and it silently changes results.

So instead of a black box, we give a **simple, explicit switch** while still
running the experiments for you.

---

## 3. Why We Chose These Technologies

| Technology | Why |
| --- | --- |
| **Scikit-learn** | The de-facto ML library; provides all models/scalers/pipelines |
| **Pandas** | DataFrames for readable leaderboards & result formatting |
| **NumPy** | Fast numeric arrays for train/test data |
| **Joblib** | Easy parallel training (`n_jobs`) with minimal code |
| **XGBoost** *(optional)* | A powerful, popular boosting library; kept optional |

**Why not a web API?** The main goal is to reduce repetitive ML *code*. A plain
Python library is the simplest, most reusable form — no server, no serialization
overhead — so it fits naturally in any notebook or script.

---

## 4. Why Each Function Exists

### `compare_models(...)` — the public face

**Why:** Users want one function that does everything: detect problem type,
run all experiments, and show a leaderboard. It's the convenience API.

### `run_experiments(...)` — the engine

**Why:** Separates the "heavy lifting" (training every combination) from the
"presentation" (building a DataFrame). This separation lets power users get the
**raw results** (list of dicts) without printing, enabling custom analysis,
saving results, or building their own reports.

### `build_leaderboard(...)` — presentation

**Why:** Turns the raw experiment results into a **sorted, readable table**
returned as a `DataFrame`. Keeping it as a pure function means you can re-sort
or reformat without re-running any model.

### `is_classification(y)` — validation

**Why:** `is_classification` is now a **required** argument to
`compare_models`/`run_experiments` — the library no longer auto-detects. But
users can still mislabel their data, so this helper validates the flag: if
`is_classification=True` is passed to regression data (or vice-versa), the
library raises a clear error instead of silently training the wrong models and
producing misleading results.

### `resolve_models(...)` — flexibility

**Why:** Users think in model *names* ("I want XGBoost"), not dictionary keys.
This helper normalises strings, lists, classes, and dicts into one internal
format, so the API feels natural no matter how you specify your models.

### `get_model_names(...)` / `get_scaler_names(...)` — discoverability

**Why:** Instead of forcing users to memorise model names, these return what's
available. Especially important because XGBoost may or may not be installed.

### `_build_preprocessor(...)` — correctness

**Why:** Wraps sklearn's `ColumnTransformer` so that only the chosen columns
are scaled and everything else passes through unchanged — making
`scale_columns` both simple and correct.

---

## 5. Why We Made These Specific Decisions

### (a) Default sort: Accuracy (classification), R² (regression)

- **Accuracy** is the most intuitive "how good is my model" number — right for
  balanced classification.
- **R²** is the most interpretable regression score (fraction of variance
  explained). We sort by R² rather than MSE so that *higher = better* works out
  of the box with descending order.

### (b) Allow `sort_by="F1"`

Many datasets are **imbalanced**; Accuracy alone is misleading there. Letting
the user sort by F1 (or anything else) makes the leaderboard meaningful for
their specific problem.

### (c) `ascending=True` for lower-is-better metrics

Regression metrics MAE/MSE/RMSE are **lower = better**, which conflicts with
descending sort. Exposing `ascending` resolves this cleanly.

### (d) Graceful failure instead of crashing

Some (model, scaler) pairs simply can't work (e.g. MultinomialNB + negative
values). We record the error and **keep going**, so one failure doesn't kill
your whole comparison.

### (e) XGBoost optional, not required

XGBoost is powerful but large and sometimes unavailable (environment
restrictions). Making it optional keeps the library lightweight while still
supporting it when present.

### (f) Parallel training built in (`n_jobs`)

Experiment grids can be big (160+ runs). `n_jobs` gives a one-line speedup for
free.

---

## 6. What This Solves for the User

```text
BEFORE (manual)                          AFTER (this library)
────────────────                              ─────────────────
Pick model                           compare_models(X_train, y_train,
Pick scaler                                  X_test, y_test)
Pick columns                                → returns leaderboard
Transform data
Train
Predict
Compute metrics
Repeat for every combination
```

The user still decides the important things (which columns to scale, which
metric matters, which models to compare), but the **repetition is automated**.

---

## 7. Limitations & Honest Trade-offs

| Limitation | Why it exists | Future fix |
| --- | --- | --- |
| No cross-validation | Keeps v1 simple; you control the split | Planned |
| Default hyperparameters only | Hyperparameter search is heavy | Planned (tuning) |
| Numeric data assumed | Categorical encoding not built in | Planned |
| No missing-value handling | Out of scope for v1 | Planned |
| XGBoost needs install | Keeps dependency light | Optional extra |

---

## 8. The Big-Picture "Why"

Machine Learning experimentation should be **fast, fair, and reproducible**.
This library takes the most repetitive part of that loop — trying many models
and scalers — and turns it into a **single call with a clear result**: a
leaderboard the user can trust, with full visibility into *how* each result was
achieved (which scaler, which columns).

It's built for the Python/ML community as **open source**, so anyone can extend
it, audit it, or add their own models, scalers, and metrics.
