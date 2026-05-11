"""Tests for rgegg.rangayyan_egg."""
import numpy as np
import pytest
from morie.fn.rgegg import rangayyan_egg


def test_rgegg_basic():
    """Test basic functionality."""
    egg = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_egg(egg, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgegg_edge():
    """Test edge cases."""
    egg = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_egg(egg, fs)
    assert isinstance(result, dict)
