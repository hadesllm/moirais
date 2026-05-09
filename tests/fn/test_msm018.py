"""Tests for msm018.mvsml_linear_mixed_models_eq_5_4."""
import numpy as np
import pytest
from moirais.fn.msm018 import mvsml_linear_mixed_models_eq_5_4


def test_msm018_basic():
    """Test basic functionality."""
    marker = np.random.default_rng(42).normal(0, 1, 100)
    information = np.random.default_rng(42).normal(0, 1, 100)
    prediction = np.random.default_rng(42).normal(0, 1, 100)
    although = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    could = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_4(marker, information, prediction, although, this, could)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm018_edge():
    """Test edge cases."""
    marker = np.random.default_rng(42).normal(0, 1, 100)
    information = np.random.default_rng(42).normal(0, 1, 100)
    prediction = np.random.default_rng(42).normal(0, 1, 100)
    although = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    could = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_4(marker, information, prediction, although, this, could)
    assert isinstance(result, dict)
