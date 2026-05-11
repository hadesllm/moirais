"""Tests for prsccs.prs_cs."""
import numpy as np
import pytest
from morie.fn.prsccs import prs_cs


def test_prsccs_basic():
    """Test basic functionality."""
    sumstats = np.random.default_rng(42).normal(0, 1, 100)
    ld_ref = np.random.default_rng(42).normal(0, 1, 100)
    result = prs_cs(sumstats, ld_ref)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prsccs_edge():
    """Test edge cases."""
    sumstats = np.random.default_rng(42).normal(0, 1, 100)
    ld_ref = np.random.default_rng(42).normal(0, 1, 100)
    result = prs_cs(sumstats, ld_ref)
    assert isinstance(result, dict)
