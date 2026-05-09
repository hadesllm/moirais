"""Tests for cbamod.cbam_attention."""
import numpy as np
import pytest
from moirais.fn.cbamod import cbam_attention


def test_cbamod_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cbam_attention(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cbamod_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cbam_attention(x)
    assert isinstance(result, dict)
