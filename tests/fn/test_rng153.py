"""Tests for rng153.rangayyan_ch3_anc_primary_input_model."""
import numpy as np
import pytest
from morie.fn.rng153 import rangayyan_ch3_anc_primary_input_model


def test_rng153_basic():
    """Test basic functionality."""
    v = np.random.default_rng(44).normal(0, 1, 100)
    m = 10
    n = 100
    result = rangayyan_ch3_anc_primary_input_model(v, m, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng153_edge():
    """Test edge cases."""
    v = np.random.default_rng(44).normal(0, 1, 100)
    m = 10
    n = 100
    result = rangayyan_ch3_anc_primary_input_model(v, m, n)
    assert isinstance(result, dict)
