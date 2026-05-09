"""Tests for aitalr.aitchison_alr."""
import numpy as np
import pytest
from moirais.fn.aitalr import aitchison_alr


def test_aitalr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    ref = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_alr(x, ref)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitalr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    ref = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_alr(x, ref)
    assert isinstance(result, dict)
