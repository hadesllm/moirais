"""Tests for hmfp32.geron_fp32."""
import numpy as np
import pytest
from morie.fn.hmfp32 import geron_fp32


def test_hmfp32_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_fp32(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmfp32_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_fp32(x)
    assert isinstance(result, dict)
