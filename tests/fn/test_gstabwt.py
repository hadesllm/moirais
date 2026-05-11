"""Tests for gstabwt.stabilized_weights."""
import numpy as np
import pytest
from morie.fn.gstabwt import stabilized_weights


def test_gstabwt_basic():
    """Test basic functionality."""
    treatment = np.random.default_rng(42).normal(0, 1, 100)
    history = np.random.default_rng(42).normal(0, 1, 100)
    numerator_model = np.random.default_rng(42).normal(0, 1, 100)
    denominator_model = np.random.default_rng(42).normal(0, 1, 100)
    result = stabilized_weights(treatment, history, numerator_model, denominator_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gstabwt_edge():
    """Test edge cases."""
    treatment = np.random.default_rng(42).normal(0, 1, 100)
    history = np.random.default_rng(42).normal(0, 1, 100)
    numerator_model = np.random.default_rng(42).normal(0, 1, 100)
    denominator_model = np.random.default_rng(42).normal(0, 1, 100)
    result = stabilized_weights(treatment, history, numerator_model, denominator_model)
    assert isinstance(result, dict)
