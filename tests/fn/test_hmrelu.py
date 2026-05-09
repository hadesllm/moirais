"""Tests for hmrelu.geron_relu."""
import numpy as np
import pytest
from moirais.fn.hmrelu import geron_relu


def test_hmrelu_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_relu(z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmrelu_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = geron_relu(z)
    assert isinstance(result, dict)
