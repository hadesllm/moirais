"""Tests for alggap.algebraic_connectivity."""
import numpy as np
import pytest
from moirais.fn.alggap import algebraic_connectivity


def test_alggap_basic():
    """Test basic functionality."""
    G = np.eye(10)
    result = algebraic_connectivity(G)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alggap_edge():
    """Test edge cases."""
    G = np.eye(10)
    result = algebraic_connectivity(G)
    assert isinstance(result, dict)
