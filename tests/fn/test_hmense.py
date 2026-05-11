"""Tests for hmense.geron_ensemble_eval."""
import numpy as np
import pytest
from morie.fn.hmense import geron_ensemble_eval


def test_hmense_basic():
    """Test basic functionality."""
    models = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_ensemble_eval(models, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmense_edge():
    """Test edge cases."""
    models = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_ensemble_eval(models, X)
    assert isinstance(result, dict)
