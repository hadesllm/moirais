"""Tests for novlt.novelty."""
import numpy as np
import pytest
from morie.fn.novlt import novelty


def test_novlt_basic():
    """Test basic functionality."""
    item = np.random.default_rng(42).normal(0, 1, 100)
    popularity = np.random.default_rng(42).normal(0, 1, 100)
    result = novelty(item, popularity)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_novlt_edge():
    """Test edge cases."""
    item = np.random.default_rng(42).normal(0, 1, 100)
    popularity = np.random.default_rng(42).normal(0, 1, 100)
    result = novelty(item, popularity)
    assert isinstance(result, dict)
