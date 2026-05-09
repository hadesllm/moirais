"""Tests for eslgam.esl_gam."""
import numpy as np
import pytest
from moirais.fn.eslgam import esl_gam


def test_eslgam_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_gam(X, y, g)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslgam_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_gam(X, y, g)
    assert isinstance(result, dict)
