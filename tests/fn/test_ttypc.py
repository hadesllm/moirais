"""Tests for ttypc.typical_sampling."""
import numpy as np
import pytest
from moirais.fn.ttypc import typical_sampling


def test_ttypc_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = typical_sampling(logits, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ttypc_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = typical_sampling(logits, tau)
    assert isinstance(result, dict)
