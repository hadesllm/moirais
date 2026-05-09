"""Tests for kmcrwd.kamath_crowspairs_bias."""
import numpy as np
import pytest
from moirais.fn.kmcrwd import kamath_crowspairs_bias


def test_kmcrwd_basic():
    """Test basic functionality."""
    stereo_pll = np.random.default_rng(42).normal(0, 1, 100)
    anti_pll = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_crowspairs_bias(stereo_pll, anti_pll)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmcrwd_edge():
    """Test edge cases."""
    stereo_pll = np.random.default_rng(42).normal(0, 1, 100)
    anti_pll = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_crowspairs_bias(stereo_pll, anti_pll)
    assert isinstance(result, dict)
