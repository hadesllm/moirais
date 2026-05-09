"""Tests for grfad.geron_forward_mode_autodiff."""
import numpy as np
import pytest
from moirais.fn.grfad import geron_forward_mode_autodiff


def test_grfad_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    x_prime = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_forward_mode_autodiff(x, x_prime, f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grfad_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    x_prime = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_forward_mode_autodiff(x, x_prime, f)
    assert isinstance(result, dict)
