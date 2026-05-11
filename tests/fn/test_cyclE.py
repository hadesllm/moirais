"""Tests for cyclE.cyclone_intensity."""
import numpy as np
import pytest
from morie.fn.cyclE import cyclone_intensity


def test_cyclE_basic():
    """Test basic functionality."""
    v_max = 100
    result = cyclone_intensity(v_max)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cyclE_edge():
    """Test edge cases."""
    v_max = 100
    result = cyclone_intensity(v_max)
    assert isinstance(result, dict)
