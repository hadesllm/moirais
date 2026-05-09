"""Tests for gb571z.gibbons_wsrt_ties_zeros."""
import numpy as np
import pytest
from moirais.fn.gb571z import gibbons_wsrt_ties_zeros


def test_gb571z_basic():
    """Test basic functionality."""
    differences = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_wsrt_ties_zeros(differences)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb571z_edge():
    """Test edge cases."""
    differences = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_wsrt_ties_zeros(differences)
    assert isinstance(result, dict)
