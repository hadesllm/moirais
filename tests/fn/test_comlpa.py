"""Tests for comlpa.label_propagation."""
import numpy as np
import pytest
from moirais.fn.comlpa import label_propagation


def test_comlpa_basic():
    """Test basic functionality."""
    G = np.eye(10)
    result = label_propagation(G)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_comlpa_edge():
    """Test edge cases."""
    G = np.eye(10)
    result = label_propagation(G)
    assert isinstance(result, dict)
