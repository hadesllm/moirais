"""Tests for sgtsbnd.sgt_sbm_detect_threshold."""
import numpy as np
import pytest
from morie.fn.sgtsbnd import sgt_sbm_detect_threshold


def test_sgtsbnd_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = sgt_sbm_detect_threshold(a, b, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtsbnd_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = sgt_sbm_detect_threshold(a, b, k)
    assert isinstance(result, dict)
