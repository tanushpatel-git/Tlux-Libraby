"""Tests for the tlux module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from tlux import (
    compare_models,
    run_experiments,
    is_classification,
    build_leaderboard,
    print_leaderboard,
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    get_model_names,
    SCALERS,
    get_scaler_names,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    return X, y


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    return X, y


@pytest.fixture
def classification_split(classification_data):
    X, y = classification_data
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def regression_split(regression_data):
    X, y = regression_data
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ---------------------------------------------------------------------------
# Test is_classification
# ---------------------------------------------------------------------------

class TestIsClassification:
    def test_numeric_with_few_unique_integers(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        assert is_classification(y) is True

    def test_numeric_with_many_values(self):
        y = np.arange(100, dtype=float)
        assert is_classification(y) is False

    def test_string_labels(self):
        y = np.array(["cat", "dog", "cat", "dog"])
        assert is_classification(y) is True

    def test_boolean_labels(self):
        y = np.array([True, False, True, False])
        assert is_classification(y) is True

    def test_float_continuous(self):
        y = np.random.randn(100)
        assert is_classification(y) is False

    def test_pandas_series(self):
        y = pd.Series([0, 1, 0, 1, 0, 1])
        assert is_classification(y) is True


# ---------------------------------------------------------------------------
# Test model and scaler catalogues
# ---------------------------------------------------------------------------

class TestCatalogues:
    def test_classification_models_not_empty(self):
        assert len(CLASSIFICATION_MODELS) > 0

    def test_regression_models_not_empty(self):
        assert len(REGRESSION_MODELS) > 0

    def test_scalers_not_empty(self):
        assert len(SCALERS) > 0

    def test_get_model_names_classification(self):
        names = get_model_names("classification")
        assert isinstance(names, list)
        assert "Random Forest" in names

    def test_get_model_names_regression(self):
        names = get_model_names("regression")
        assert isinstance(names, list)
        assert "Random Forest" in names

    def test_get_scaler_names(self):
        names = get_scaler_names()
        assert isinstance(names, list)
        assert "No Scaling" in names
        assert "StandardScaler" in names

    def test_get_scaler_names_no_no_scaling(self):
        names = get_scaler_names(include_no_scaling=False)
        assert "No Scaling" not in names
        assert "StandardScaler" in names


# ---------------------------------------------------------------------------
# Test run_experiments
# ---------------------------------------------------------------------------

class TestRunExperiments:
    def test_classification_single_model(self, classification_split):
        X_train, X_test, y_train, y_test = classification_split
        results = run_experiments(
            X_train, y_train, X_test, y_test,
            models={"RF": RandomForestClassifier},
            scalers=["No Scaling"],
            verbose=False,
        )
        assert len(results) == 1
        assert results[0]["model_name"] == "RF"
        assert results[0]["error"] is None
        assert "Accuracy" in results[0]["metrics"]

    def test_regression_single_model(self, regression_split):
        X_train, X_test, y_train, y_test = regression_split
        results = run_experiments(
            X_train, y_train, X_test, y_test,
            models={"RF": RandomForestRegressor},
            scalers=["No Scaling"],
            verbose=False,
        )
        assert len(results) == 1
        assert results[0]["model_name"] == "RF"
        assert results[0]["error"] is None
        assert "R2" in results[0]["metrics"]

    def test_multiple_scalers(self, classification_split):
        X_train, X_test, y_train, y_test = classification_split
        results = run_experiments(
            X_train, y_train, X_test, y_test,
            models={"LR": LogisticRegression},
            scalers=["No Scaling", "StandardScaler"],
            verbose=False,
        )
        assert len(results) == 2
        scaler_names = {r["scaler_name"] for r in results}
        assert "No Scaling" in scaler_names
        assert "StandardScaler" in scaler_names

    def test_auto_problem_detection(self, classification_split):
        X_train, X_test, y_train, y_test = classification_split
        results = run_experiments(
            X_train, y_train, X_test, y_test,
            models={"LR": LogisticRegression},
            scalers=["No Scaling"],
            verbose=False,
        )
        assert len(results) == 1
        assert results[0]["error"] is None

    def test_custom_metrics(self, classification_split):
        X_train, X_test, y_train, y_test = classification_split
        from sklearn.metrics import accuracy_score
        custom = {"Acc": accuracy_score}
        results = run_experiments(
            X_train, y_train, X_test, y_test,
            models={"LR": LogisticRegression},
            scalers=["No Scaling"],
            metrics=custom,
            verbose=False,
        )
        assert "Acc" in results[0]["metrics"]

    def test_empty_models_list_raises(self, classification_split):
        X_train, X_test, y_train, y_test = classification_split
        with pytest.raises(ValueError, match="Unknown model"):
            run_experiments(
                X_train, y_train, X_test, y_test,
                models=["NonExistentModel"],
                scalers=["No Scaling"],
                verbose=False,
            )


# ---------------------------------------------------------------------------
# Test build_leaderboard
# ---------------------------------------------------------------------------

class TestBuildLeaderboard:
    def test_empty_results(self):
        df = build_leaderboard([])
        assert df.empty

    def test_leaderboard_structure(self, classification_split):
        X_train, X_test, y_train, y_test = classification_split
        results = run_experiments(
            X_train, y_train, X_test, y_test,
            models={"LR": LogisticRegression},
            scalers=["No Scaling"],
            verbose=False,
        )
        df = build_leaderboard(results)
        assert "Model" in df.columns
        assert "Scaling" in df.columns
        assert "Accuracy" in df.columns

    def test_leaderboard_sorting(self, classification_split):
        X_train, X_test, y_train, y_test = classification_split
        results = run_experiments(
            X_train, y_train, X_test, y_test,
            models={"LR": LogisticRegression},
            scalers=["No Scaling"],
            verbose=False,
        )
        df = build_leaderboard(results, sort_by="Accuracy", ascending=False)
        assert df["Accuracy"].is_monotonic_decreasing

    def test_leaderboard_invalid_sort(self, classification_split):
        X_train, X_test, y_train, y_test = classification_split
        results = run_experiments(
            X_train, y_train, X_test, y_test,
            models={"LR": LogisticRegression},
            scalers=["No Scaling"],
            verbose=False,
        )
        with pytest.raises(ValueError, match="sort_by="):
            build_leaderboard(results, sort_by="NonExistent")


# ---------------------------------------------------------------------------
# Test compare_models (high-level API)
# ---------------------------------------------------------------------------

class TestCompareModels:
    def test_returns_dataframe(self, classification_split):
        X_train, X_test, y_train, y_test = classification_split
        df = compare_models(
            X_train, y_train, X_test, y_test,
            models={"LR": LogisticRegression},
            scalers=["No Scaling"],
            verbose=False,
        )
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_regression(self, regression_split):
        X_train, X_test, y_train, y_test = regression_split
        df = compare_models(
            X_train, y_train, X_test, y_test,
            models={"LR": LinearRegression},
            scalers=["No Scaling"],
            verbose=False,
        )
        assert isinstance(df, pd.DataFrame)
        assert "R2" in df.columns

    def test_detailed_output(self, classification_split, capsys):
        X_train, X_test, y_train, y_test = classification_split
        compare_models(
            X_train, y_train, X_test, y_test,
            models={"LR": LogisticRegression},
            scalers=["No Scaling"],
            verbose=False,
            detailed=True,
        )
        captured = capsys.readouterr()
        assert "Detailed Results" in captured.out

    def test_with_dataframe_input(self, classification_split):
        X_train, X_test, y_train, y_test = classification_split
        df_compare = compare_models(
            pd.DataFrame(X_train), pd.Series(y_train),
            pd.DataFrame(X_test), pd.Series(y_test),
            models={"LR": LogisticRegression},
            scalers=["No Scaling"],
            verbose=False,
        )
        assert isinstance(df_compare, pd.DataFrame)
        assert not df_compare.empty
