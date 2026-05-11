"""Tests for hmfp16.geron_fp16_quant."""
import numpy as np
import pytest
from morie.fn.hmfp16 import geron_fp16_quant


def test_hmfp16_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_fp16_quant(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmfp16_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_fp16_quant(x)
    assert isinstance(result, dict)
