"""Tests for ssmpar.ssm_parallel_scan."""
import numpy as np
import pytest
from moirais.fn.ssmpar import ssm_parallel_scan


def test_ssmpar_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = ssm_parallel_scan(y, A, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ssmpar_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = ssm_parallel_scan(y, A, B)
    assert isinstance(result, dict)
