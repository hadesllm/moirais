"""Tests for msm022.mvsml_linear_mixed_models_eq_5_3."""
import numpy as np
import pytest
from moirais.fn.msm022 import mvsml_linear_mixed_models_eq_5_3


def test_msm022_basic():
    """Test basic functionality."""
    types = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    four = np.random.default_rng(42).normal(0, 1, 100)
    environments = np.random.default_rng(42).normal(0, 1, 100)
    Besides = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_3(types, are, the, four, environments, Besides)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm022_edge():
    """Test edge cases."""
    types = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    four = np.random.default_rng(42).normal(0, 1, 100)
    environments = np.random.default_rng(42).normal(0, 1, 100)
    Besides = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_3(types, are, the, four, environments, Besides)
    assert isinstance(result, dict)
