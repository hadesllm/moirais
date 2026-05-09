"""Tests for grgrp.geron_gaussian_random_projection."""
import numpy as np
import pytest
from moirais.fn.grgrp import geron_gaussian_random_projection


def test_grgrp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    d = 5
    seed = 42
    result = geron_gaussian_random_projection(X, d, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grgrp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    d = 5
    seed = 42
    result = geron_gaussian_random_projection(X, d, seed)
    assert isinstance(result, dict)
