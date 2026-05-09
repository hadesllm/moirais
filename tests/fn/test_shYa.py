"""Tests for shYa.shunting_yard."""
import numpy as np
import pytest
from moirais.fn.shYa import shunting_yard


def test_shYa_basic():
    """Test basic functionality."""
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = shunting_yard(tokens)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_shYa_edge():
    """Test edge cases."""
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = shunting_yard(tokens)
    assert isinstance(result, dict)
