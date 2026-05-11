"""Tests for midor.model_identify_estimate_refute."""
import numpy as np
import pytest
from morie.fn.midor import model_identify_estimate_refute


def test_midor_basic():
    """Test basic functionality."""
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    data = np.random.default_rng(42).normal(0, 1, 100)
    estimand = np.random.default_rng(42).normal(0, 1, 100)
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    result = model_identify_estimate_refute(dag, data, estimand, estimator)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_midor_edge():
    """Test edge cases."""
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    data = np.random.default_rng(42).normal(0, 1, 100)
    estimand = np.random.default_rng(42).normal(0, 1, 100)
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    result = model_identify_estimate_refute(dag, data, estimand, estimator)
    assert isinstance(result, dict)
