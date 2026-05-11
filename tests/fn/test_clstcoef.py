"""Tests for clstcoef.clustering_coefficient."""
import numpy as np
import pytest
from morie.fn.clstcoef import clustering_coefficient


def test_clstcoef_basic():
    """Test basic functionality."""
    G = np.eye(10)
    result = clustering_coefficient(G)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_clstcoef_edge():
    """Test edge cases."""
    G = np.eye(10)
    result = clustering_coefficient(G)
    assert isinstance(result, dict)
