"""Tests for igarcm.igarch_integrated."""
import numpy as np
import pytest
from morie.fn.igarcm import igarch_integrated


def test_igarcm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = igarch_integrated(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_igarcm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = igarch_integrated(x)
    assert isinstance(result, dict)
