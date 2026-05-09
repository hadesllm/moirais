"""Tests for hml1c.geron_one_cycle."""
import numpy as np
import pytest
from moirais.fn.hml1c import geron_one_cycle


def test_hml1c_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    lr_max = 100
    lr_min = 0
    result = geron_one_cycle(t, T, lr_max, lr_min)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hml1c_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    lr_max = 100
    lr_min = 0
    result = geron_one_cycle(t, T, lr_max, lr_min)
    assert isinstance(result, dict)
