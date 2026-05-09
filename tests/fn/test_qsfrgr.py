"""Tests for qsfrgr.quantile_survival_forest."""
import numpy as np
import pytest
from moirais.fn.qsfrgr import quantile_survival_forest


def test_qsfrgr_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    quantile = np.random.default_rng(42).normal(0, 1, 100)
    result = quantile_survival_forest(time, event, X, quantile)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_qsfrgr_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    quantile = np.random.default_rng(42).normal(0, 1, 100)
    result = quantile_survival_forest(time, event, X, quantile)
    assert isinstance(result, dict)
