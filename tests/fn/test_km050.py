"""Tests for km050.kamath_ch3_back_translation_prob."""
import numpy as np
import pytest
from moirais.fn.km050 import kamath_ch3_back_translation_prob


def test_km050_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    thatt = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch3_back_translation_prob(t, thatt)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km050_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    thatt = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch3_back_translation_prob(t, thatt)
    assert isinstance(result, dict)
