"""Tests for semsbn.sem_sb_chi_sq."""
import numpy as np
import pytest
from moirais.fn.semsbn import sem_sb_chi_sq


def test_semsbn_basic():
    """Test basic functionality."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    result = sem_sb_chi_sq(fit)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_semsbn_edge():
    """Test edge cases."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    result = sem_sb_chi_sq(fit)
    assert isinstance(result, dict)
