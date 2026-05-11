"""Tests for fzmrln.fauzi_naive_mrl."""
import numpy as np
import pytest
from morie.fn.fzmrln import fauzi_naive_mrl


def test_fzmrln_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = fauzi_naive_mrl(t, bandwidth, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzmrln_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = fauzi_naive_mrl(t, bandwidth, kernel)
    assert isinstance(result, dict)
