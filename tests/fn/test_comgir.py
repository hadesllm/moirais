"""Tests for comgir.girvan_newman."""
import numpy as np
import pytest
from moirais.fn.comgir import girvan_newman


def test_comgir_basic():
    """Test basic functionality."""
    G = np.eye(10)
    result = girvan_newman(G)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_comgir_edge():
    """Test edge cases."""
    G = np.eye(10)
    result = girvan_newman(G)
    assert isinstance(result, dict)
