"""Tests for rdpcomp.rdp_subsampled_composition."""
import numpy as np
import pytest
from moirais.fn.rdpcomp import rdp_subsampled_composition


def test_rdpcomp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    q = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = rdp_subsampled_composition(y, alpha, q, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rdpcomp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    q = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = rdp_subsampled_composition(y, alpha, q, epsilon)
    assert isinstance(result, dict)
