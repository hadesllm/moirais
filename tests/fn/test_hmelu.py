"""Tests for hmelu.geron_elu."""
import numpy as np
import pytest
from moirais.fn.hmelu import geron_elu


def test_hmelu_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    alpha = 0.05
    result = geron_elu(z, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmelu_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    alpha = 0.05
    result = geron_elu(z, alpha)
    assert isinstance(result, dict)
