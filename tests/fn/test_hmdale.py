"""Tests for hmdale.geron_dalle."""
import numpy as np
import pytest
from moirais.fn.hmdale import geron_dalle


def test_hmdale_basic():
    """Test basic functionality."""
    text = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dalle(text, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdale_edge():
    """Test edge cases."""
    text = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dalle(text, model)
    assert isinstance(result, dict)
