"""Tests for hmhei.geron_he_init."""
import numpy as np
import pytest
from morie.fn.hmhei import geron_he_init


def test_hmhei_basic():
    """Test basic functionality."""
    fan_in = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_he_init(fan_in, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmhei_edge():
    """Test edge cases."""
    fan_in = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_he_init(fan_in, seed)
    assert isinstance(result, dict)
