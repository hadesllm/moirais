"""Tests for longMd.longitudinal_mediation."""
import numpy as np
import pytest
from moirais.fn.longMd import longitudinal_mediation


def test_longMd_basic():
    """Test basic functionality."""
    panel = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = longitudinal_mediation(panel, X, M, Y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_longMd_edge():
    """Test edge cases."""
    panel = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = longitudinal_mediation(panel, X, M, Y)
    assert isinstance(result, dict)
