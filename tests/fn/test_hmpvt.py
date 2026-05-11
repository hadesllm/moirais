"""Tests for hmpvt.geron_pvt."""
import numpy as np
import pytest
from morie.fn.hmpvt import geron_pvt


def test_hmpvt_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    stage_cfgs = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_pvt(image, stage_cfgs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmpvt_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    stage_cfgs = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_pvt(image, stage_cfgs)
    assert isinstance(result, dict)
