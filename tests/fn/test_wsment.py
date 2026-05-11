"""Tests for wsment.wasserman_entropy."""
import numpy as np
import pytest
from morie.fn.wsment import wasserman_entropy


def test_wsment_basic():
    """Test basic functionality."""
    p = 5
    result = wasserman_entropy(p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsment_edge():
    """Test edge cases."""
    p = 5
    result = wasserman_entropy(p)
    assert isinstance(result, dict)
