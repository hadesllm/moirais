"""Tests for hrzcs.horowitz_curse_dimensionality."""
import numpy as np
import pytest
from moirais.fn.hrzcs import horowitz_curse_dimensionality


def test_hrzcs_basic():
    """Test basic functionality."""
    d = 5
    n = 100
    result = horowitz_curse_dimensionality(d, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzcs_edge():
    """Test edge cases."""
    d = 5
    n = 100
    result = horowitz_curse_dimensionality(d, n)
    assert isinstance(result, dict)
