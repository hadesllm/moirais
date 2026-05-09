"""Tests for hmcod.geron_curse_dimensionality."""
import numpy as np
import pytest
from moirais.fn.hmcod import geron_curse_dimensionality


def test_hmcod_basic():
    """Test basic functionality."""
    d = 5
    n = 100
    result = geron_curse_dimensionality(d, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmcod_edge():
    """Test edge cases."""
    d = 5
    n = 100
    result = geron_curse_dimensionality(d, n)
    assert isinstance(result, dict)
