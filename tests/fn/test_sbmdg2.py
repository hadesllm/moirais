"""Tests for sbmdg2.degree_corrected_sbm."""
import numpy as np
import pytest
from moirais.fn.sbmdg2 import degree_corrected_sbm


def test_sbmdg2_basic():
    """Test basic functionality."""
    G = np.eye(10)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = degree_corrected_sbm(G, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sbmdg2_edge():
    """Test edge cases."""
    G = np.eye(10)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = degree_corrected_sbm(G, K)
    assert isinstance(result, dict)
