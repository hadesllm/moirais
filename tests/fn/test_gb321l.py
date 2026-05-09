"""Tests for gb321l.gibbons_distributing_objects."""
import numpy as np
import pytest
from moirais.fn.gb321l import gibbons_distributing_objects


def test_gb321l_basic():
    """Test basic functionality."""
    n = 100
    r = 10
    result = gibbons_distributing_objects(n, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb321l_edge():
    """Test edge cases."""
    n = 100
    r = 10
    result = gibbons_distributing_objects(n, r)
    assert isinstance(result, dict)
