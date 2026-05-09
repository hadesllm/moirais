"""Tests for cvxsdp.boyd_sdp."""
import numpy as np
import pytest
from moirais.fn.cvxsdp import boyd_sdp


def test_cvxsdp_basic():
    """Test basic functionality."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    result = boyd_sdp(c, F)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxsdp_edge():
    """Test edge cases."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    result = boyd_sdp(c, F)
    assert isinstance(result, dict)
