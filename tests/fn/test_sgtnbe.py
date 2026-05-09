"""Tests for sgtnbe.sgt_nonbacktracking_matrix."""
import numpy as np
import pytest
from moirais.fn.sgtnbe import sgt_nonbacktracking_matrix


def test_sgtnbe_basic():
    """Test basic functionality."""
    edges = [('A', 'B'), ('B', 'C')]
    n = 100
    result = sgt_nonbacktracking_matrix(edges, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtnbe_edge():
    """Test edge cases."""
    edges = [('A', 'B'), ('B', 'C')]
    n = 100
    result = sgt_nonbacktracking_matrix(edges, n)
    assert isinstance(result, dict)
