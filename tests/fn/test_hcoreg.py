"""Tests for hcoreg.hardcore_process."""
import numpy as np
import pytest
from morie.fn.hcoreg import hardcore_process


def test_hcoreg_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    r = 10
    lam = 0.1
    result = hardcore_process(coords, r, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hcoreg_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    r = 10
    lam = 0.1
    result = hardcore_process(coords, r, lam)
    assert isinstance(result, dict)
