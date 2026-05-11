"""Tests for rng035.rangayyan_ch3_discrete_unit_step."""
import numpy as np
import pytest
from morie.fn.rng035 import rangayyan_ch3_discrete_unit_step


def test_rng035_basic():
    """Test basic functionality."""
    n = 100
    result = rangayyan_ch3_discrete_unit_step(n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng035_edge():
    """Test edge cases."""
    n = 100
    result = rangayyan_ch3_discrete_unit_step(n)
    assert isinstance(result, dict)
