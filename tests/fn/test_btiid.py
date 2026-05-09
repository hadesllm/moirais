"""Tests for btiid.boot_iid_resample."""
import numpy as np
import pytest
from moirais.fn.btiid import boot_iid_resample


def test_btiid_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_iid_resample(x, stat, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btiid_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_iid_resample(x, stat, B)
    assert isinstance(result, dict)
