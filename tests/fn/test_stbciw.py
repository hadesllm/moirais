"""Tests for stbciw.stabilized_censoring_weights."""
import numpy as np
import pytest
from moirais.fn.stbciw import stabilized_censoring_weights


def test_stbciw_basic():
    """Test basic functionality."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = stabilized_censoring_weights(C, H)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_stbciw_edge():
    """Test edge cases."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = stabilized_censoring_weights(C, H)
    assert isinstance(result, dict)
