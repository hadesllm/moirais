"""Tests for msm142.mvsml_categorical_count_eq_8_9."""
import numpy as np
import pytest
from moirais.fn.msm142 import mvsml_categorical_count_eq_8_9


def test_msm142_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    using = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    BGLR = np.random.default_rng(42).normal(0, 1, 100)
    package = np.random.default_rng(42).normal(0, 1, 100)
    The = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_9(model, using, the, BGLR, package, The)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm142_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    using = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    BGLR = np.random.default_rng(42).normal(0, 1, 100)
    package = np.random.default_rng(42).normal(0, 1, 100)
    The = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_categorical_count_eq_8_9(model, using, the, BGLR, package, The)
    assert isinstance(result, dict)
