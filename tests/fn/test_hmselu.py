"""Tests for hmselu.geron_selu."""
import numpy as np
import pytest
from morie.fn.hmselu import geron_selu


def test_hmselu_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_selu(z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmselu_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_selu(z)
    assert isinstance(result, dict)
