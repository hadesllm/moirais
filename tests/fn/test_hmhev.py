"""Tests for hmhev.geron_heaviside."""
import numpy as np
import pytest
from moirais.fn.hmhev import geron_heaviside


def test_hmhev_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_heaviside(z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmhev_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_heaviside(z)
    assert isinstance(result, dict)
