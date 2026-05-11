"""Tests for rng034.rangayyan_ch3_discrete_delta."""
import numpy as np
import pytest
from morie.fn.rng034 import rangayyan_ch3_discrete_delta


def test_rng034_basic():
    """Test basic functionality."""
    n = 100
    result = rangayyan_ch3_discrete_delta(n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng034_edge():
    """Test edge cases."""
    n = 100
    result = rangayyan_ch3_discrete_delta(n)
    assert isinstance(result, dict)
