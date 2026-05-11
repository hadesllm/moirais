"""Tests for trnscl.transitivity."""
import numpy as np
import pytest
from morie.fn.trnscl import transitivity


def test_trnscl_basic():
    """Test basic functionality."""
    G = np.eye(10)
    result = transitivity(G)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_trnscl_edge():
    """Test edge cases."""
    G = np.eye(10)
    result = transitivity(G)
    assert isinstance(result, dict)
