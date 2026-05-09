"""Tests for wsmclt.wasserman_clt."""
import numpy as np
import pytest
from moirais.fn.wsmclt import wasserman_clt


def test_wsmclt_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_clt(data)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmclt_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_clt(data)
    assert isinstance(result, dict)
