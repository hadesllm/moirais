"""Tests for sxrhrt.sex_specific_h2."""
import numpy as np
import pytest
from moirais.fn.sxrhrt import sex_specific_h2


def test_sxrhrt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    sex = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = sex_specific_h2(y, sex, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sxrhrt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    sex = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = sex_specific_h2(y, sex, K)
    assert isinstance(result, dict)
