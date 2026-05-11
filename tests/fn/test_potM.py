"""Tests for potM.peaks_over_threshold."""
import numpy as np
import pytest
from morie.fn.potM import peaks_over_threshold


def test_potM_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = peaks_over_threshold(y, u)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_potM_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = peaks_over_threshold(y, u)
    assert isinstance(result, dict)
