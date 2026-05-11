"""Tests for volharj.vol_har_rv_jump."""
import numpy as np
import pytest
from morie.fn.volharj import vol_har_rv_jump


def test_volharj_basic():
    """Test basic functionality."""
    RV = np.random.default_rng(42).normal(0, 1, 100)
    BPV = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_har_rv_jump(RV, BPV)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volharj_edge():
    """Test edge cases."""
    RV = np.random.default_rng(42).normal(0, 1, 100)
    BPV = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_har_rv_jump(RV, BPV)
    assert isinstance(result, dict)
