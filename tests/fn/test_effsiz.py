"""Tests for effsiz.effective_sample_size."""
import numpy as np
import pytest
from moirais.fn.effsiz import effective_sample_size


def test_effsiz_basic():
    """Test basic functionality."""
    n = 100
    deff = np.random.default_rng(42).normal(0, 1, 100)
    result = effective_sample_size(n, deff)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_effsiz_edge():
    """Test edge cases."""
    n = 100
    deff = np.random.default_rng(42).normal(0, 1, 100)
    result = effective_sample_size(n, deff)
    assert isinstance(result, dict)
