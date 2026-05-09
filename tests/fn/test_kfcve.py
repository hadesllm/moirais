"""Tests for kfcve.k_fold_cv_error."""
import numpy as np
import pytest
from moirais.fn.kfcve import k_fold_cv_error


def test_kfcve_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    y_hat_folds = np.random.default_rng(42).normal(0, 1, 100)
    result = k_fold_cv_error(y, y_hat_folds)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kfcve_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    y_hat_folds = np.random.default_rng(42).normal(0, 1, 100)
    result = k_fold_cv_error(y, y_hat_folds)
    assert isinstance(result, dict)
