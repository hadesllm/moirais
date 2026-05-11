"""Tests for grhei.geron_he_init."""
import numpy as np
import pytest
from morie.fn.grhei import geron_he_init


def test_grhei_basic():
    """Test basic functionality."""
    fan_in = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_he_init(fan_in)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grhei_edge():
    """Test edge cases."""
    fan_in = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_he_init(fan_in)
    assert isinstance(result, dict)
