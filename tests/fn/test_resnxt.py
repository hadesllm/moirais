"""Tests for resnxt.resnext_block."""
import numpy as np
import pytest
from moirais.fn.resnxt import resnext_block


def test_resnxt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    cardinality = np.random.default_rng(42).normal(0, 1, 100)
    result = resnext_block(x, cardinality)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_resnxt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    cardinality = np.random.default_rng(42).normal(0, 1, 100)
    result = resnext_block(x, cardinality)
    assert isinstance(result, dict)
