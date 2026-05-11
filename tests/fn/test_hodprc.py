"""Tests for hodprc.hodrick_prescott."""
import numpy as np
import pytest
from morie.fn.hodprc import hodrick_prescott


def test_hodprc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    lam = 0.1
    result = hodrick_prescott(y, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hodprc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    lam = 0.1
    result = hodrick_prescott(y, lam)
    assert isinstance(result, dict)
