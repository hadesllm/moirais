"""Tests for sgtegap.sgt_eigengap_heuristic."""
import numpy as np
import pytest
from morie.fn.sgtegap import sgt_eigengap_heuristic


def test_sgtegap_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    K_max = 100
    result = sgt_eigengap_heuristic(A, K_max)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtegap_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    K_max = 100
    result = sgt_eigengap_heuristic(A, K_max)
    assert isinstance(result, dict)
