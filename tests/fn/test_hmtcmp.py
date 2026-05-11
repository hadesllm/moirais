"""Tests for hmtcmp.geron_torch_compile."""
import numpy as np
import pytest
from morie.fn.hmtcmp import geron_torch_compile


def test_hmtcmp_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    mode = 'auto'
    result = geron_torch_compile(model, mode)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmtcmp_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    mode = 'auto'
    result = geron_torch_compile(model, mode)
    assert isinstance(result, dict)
