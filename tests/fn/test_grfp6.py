"""Tests for grfp6.geron_fp16_mixed_precision."""
import numpy as np
import pytest
from moirais.fn.grfp6 import geron_fp16_mixed_precision


def test_grfp6_basic():
    """Test basic functionality."""
    loss = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_fp16_mixed_precision(loss, S)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grfp6_edge():
    """Test edge cases."""
    loss = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_fp16_mixed_precision(loss, S)
    assert isinstance(result, dict)
