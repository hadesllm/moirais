"""Tests for rgchoi.rangayyan_choi_williams."""
import numpy as np
import pytest
from moirais.fn.rgchoi import rangayyan_choi_williams


def test_rgchoi_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    sigma = 1.0
    result = rangayyan_choi_williams(x, fs, sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgchoi_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    sigma = 1.0
    result = rangayyan_choi_williams(x, fs, sigma)
    assert isinstance(result, dict)
