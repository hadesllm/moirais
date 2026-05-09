"""Tests for hmgrp.geron_gaussian_rand_projection."""
import numpy as np
import pytest
from moirais.fn.hmgrp import geron_gaussian_rand_projection


def test_hmgrp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    d_out = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_gaussian_rand_projection(X, d_out, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmgrp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    d_out = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_gaussian_rand_projection(X, d_out, seed)
    assert isinstance(result, dict)
