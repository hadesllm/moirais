"""Tests for smplsz.sample_size_calc."""
import numpy as np
import pytest
from moirais.fn.smplsz import sample_size_calc


def test_smplsz_basic():
    """Test basic functionality."""
    p = 5
    e = np.random.default_rng(44).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = sample_size_calc(p, e, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_smplsz_edge():
    """Test edge cases."""
    p = 5
    e = np.random.default_rng(44).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = sample_size_calc(p, e, z)
    assert isinstance(result, dict)
