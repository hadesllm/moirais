"""Tests for morie.fn.robust — Random Forest robustness evaluation."""

import pytest
import pandas as pd
from sklearn.datasets import make_classification
from morie.fn.robust import eval_robustness as robust


@pytest.fixture()
def binary_data():
    """Simple binary classification dataset."""
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=3,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    y_sr = pd.Series(y, name="target")
    return X_df[:150], y_sr[:150], X_df[150:], y_sr[150:]


class TestEvalRobustness:
    """Tests for eval_robustness."""

    def test_returns_dict(self, binary_data):
        """Result should be a dict."""
        X_train, y_train, X_test, y_test = binary_data
        result = robust(X_train, y_train, X_test, y_test)
        assert isinstance(result, dict)

    def test_has_accuracy_key(self, binary_data):
        """Result dict should contain 'accuracy' key."""
        X_train, y_train, X_test, y_test = binary_data
        result = robust(X_train, y_train, X_test, y_test)
        assert "accuracy" in result

    def test_accuracy_range(self, binary_data):
        """Accuracy should be between 0 and 1."""
        X_train, y_train, X_test, y_test = binary_data
        result = robust(X_train, y_train, X_test, y_test)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_has_class_reports(self, binary_data):
        """Should have per-class precision/recall/f1."""
        X_train, y_train, X_test, y_test = binary_data
        result = robust(X_train, y_train, X_test, y_test)
        # sklearn classification_report has keys "0" and "1" for binary
        assert "0" in result or 0 in result
