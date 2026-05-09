"""Tests for pinsk1.pinsker_inequality."""
import numpy as np
import pytest
from moirais.fn.pinsk1 import pinsker_inequality


def test_pinsk1_basic():
    """Test basic functionality."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = pinsker_inequality(p, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pinsk1_edge():
    """Test edge cases."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = pinsker_inequality(p, q)
    assert isinstance(result, dict)
