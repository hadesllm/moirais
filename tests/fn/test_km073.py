"""Tests for km073.kamath_ch5_pref_sigmoid_form."""
import numpy as np
import pytest
from moirais.fn.km073 import kamath_ch5_pref_sigmoid_form


def test_km073_basic():
    """Test basic functionality."""
    r_star = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch5_pref_sigmoid_form(r_star)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km073_edge():
    """Test edge cases."""
    r_star = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch5_pref_sigmoid_form(r_star)
    assert isinstance(result, dict)
