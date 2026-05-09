"""Tests for mutifo.mutual_information."""
import numpy as np
import pytest
from moirais.fn.mutifo import mutual_information


def test_mutifo_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y2 = np.random.default_rng(42).normal(0, 1, 100)
    result = mutual_information(y, x, y2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mutifo_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y2 = np.random.default_rng(42).normal(0, 1, 100)
    result = mutual_information(y, x, y2)
    assert isinstance(result, dict)
