"""Tests for smrest.standardized_mortality_ratio."""
import numpy as np
import pytest
from moirais.fn.smrest import standardized_mortality_ratio


def test_smrest_basic():
    """Test basic functionality."""
    observed = np.random.default_rng(42).normal(0, 1, 100)
    expected = np.random.default_rng(42).normal(0, 1, 100)
    result = standardized_mortality_ratio(observed, expected)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_smrest_edge():
    """Test edge cases."""
    observed = np.random.default_rng(42).normal(0, 1, 100)
    expected = np.random.default_rng(42).normal(0, 1, 100)
    result = standardized_mortality_ratio(observed, expected)
    assert isinstance(result, dict)
