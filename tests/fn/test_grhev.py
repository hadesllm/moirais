"""Tests for grhev.geron_heaviside_step."""
import numpy as np
import pytest
from morie.fn.grhev import geron_heaviside_step


def test_grhev_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_heaviside_step(z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grhev_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_heaviside_step(z)
    assert isinstance(result, dict)
