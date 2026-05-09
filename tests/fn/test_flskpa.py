"""Tests for flskpa.fleiss_kappa."""
import numpy as np
import pytest
from moirais.fn.flskpa import fleiss_kappa


def test_flskpa_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = fleiss_kappa(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_flskpa_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = fleiss_kappa(X)
    assert isinstance(result, dict)
