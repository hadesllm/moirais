"""Tests for aitclr.aitchison_clr."""
import numpy as np
import pytest
from morie.fn.aitclr import aitchison_clr


def test_aitclr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_clr(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitclr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_clr(x)
    assert isinstance(result, dict)
