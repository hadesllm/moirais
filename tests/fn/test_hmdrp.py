"""Tests for hmdrp.geron_dropout."""
import numpy as np
import pytest
from moirais.fn.hmdrp import geron_dropout


def test_hmdrp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    training = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dropout(x, p, training)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdrp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    training = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dropout(x, p, training)
    assert isinstance(result, dict)
