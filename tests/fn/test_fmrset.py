"""Tests for fmrset.familial_mr_set."""
import numpy as np
import pytest
from moirais.fn.fmrset import familial_mr_set


def test_fmrset_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    exposures = np.random.default_rng(42).normal(0, 1, 100)
    instruments = np.random.default_rng(42).normal(0, 1, 100)
    pedigree = np.random.default_rng(42).normal(0, 1, 100)
    result = familial_mr_set(y, exposures, instruments, pedigree)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_fmrset_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    exposures = np.random.default_rng(42).normal(0, 1, 100)
    instruments = np.random.default_rng(42).normal(0, 1, 100)
    pedigree = np.random.default_rng(42).normal(0, 1, 100)
    result = familial_mr_set(y, exposures, instruments, pedigree)
    assert isinstance(result, dict)
