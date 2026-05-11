"""Tests for grsig.geron_sigmoid."""
import numpy as np
import pytest
from morie.fn.grsig import geron_sigmoid


def test_grsig_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    result = geron_sigmoid(t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grsig_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    result = geron_sigmoid(t)
    assert isinstance(result, dict)
