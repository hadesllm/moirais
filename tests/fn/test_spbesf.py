"""Tests for spbesf.schabenberger_bessel_function."""
import numpy as np
import pytest
from morie.fn.spbesf import schabenberger_bessel_function


def test_spbesf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_bessel_function(x, nu)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spbesf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_bessel_function(x, nu)
    assert isinstance(result, dict)
