"""Tests for ldprun.ld_pruning."""
import numpy as np
import pytest
from moirais.fn.ldprun import ld_pruning


def test_ldprun_basic():
    """Test basic functionality."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    r2_threshold = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = ld_pruning(genotypes, r2_threshold, window)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ldprun_edge():
    """Test edge cases."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    r2_threshold = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = ld_pruning(genotypes, r2_threshold, window)
    assert isinstance(result, dict)
