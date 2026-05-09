"""Tests for nrmprc.normalized_inverse_gauss."""
import numpy as np
import pytest
from moirais.fn.nrmprc import normalized_inverse_gauss


def test_nrmprc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    tau = 0.1
    result = normalized_inverse_gauss(y, alpha, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_nrmprc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    tau = 0.1
    result = normalized_inverse_gauss(y, alpha, tau)
    assert isinstance(result, dict)
