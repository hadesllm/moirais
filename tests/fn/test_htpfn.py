"""Tests for htpfn.htp_functional_predictor."""
import numpy as np
import pytest
from morie.fn.htpfn import htp_functional_predictor


def test_htpfn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    W_functional = np.random.default_rng(42).normal(0, 1, 100)
    result = htp_functional_predictor(y, markers, W_functional)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_htpfn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    W_functional = np.random.default_rng(42).normal(0, 1, 100)
    result = htp_functional_predictor(y, markers, W_functional)
    assert isinstance(result, dict)
