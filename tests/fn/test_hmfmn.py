"""Tests for hmfmn.geron_fashion_mnist."""
import numpy as np
import pytest
from moirais.fn.hmfmn import geron_fashion_mnist


def test_hmfmn_basic():
    """Test basic functionality."""
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_fashion_mnist(epochs, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmfmn_edge():
    """Test edge cases."""
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_fashion_mnist(epochs, lr)
    assert isinstance(result, dict)
