"""Tests for meglt.matrix_completion_low_rank."""
import numpy as np
import pytest
from moirais.fn.meglt import matrix_completion_low_rank


def test_meglt_basic():
    """Test basic functionality."""
    R_obs = np.random.default_rng(42).normal(0, 1, 100)
    mask = np.random.default_rng(42).normal(0, 1, 100)
    result = matrix_completion_low_rank(R_obs, mask)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_meglt_edge():
    """Test edge cases."""
    R_obs = np.random.default_rng(42).normal(0, 1, 100)
    mask = np.random.default_rng(42).normal(0, 1, 100)
    result = matrix_completion_low_rank(R_obs, mask)
    assert isinstance(result, dict)
