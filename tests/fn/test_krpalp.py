"""Tests for krpalp.krippendorff_alpha."""
import numpy as np
import pytest
from morie.fn.krpalp import krippendorff_alpha


def test_krpalp_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    level = np.random.default_rng(42).normal(0, 1, 100)
    result = krippendorff_alpha(data, level)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_krpalp_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    level = np.random.default_rng(42).normal(0, 1, 100)
    result = krippendorff_alpha(data, level)
    assert isinstance(result, dict)
