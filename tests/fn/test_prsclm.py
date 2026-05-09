"""Tests for prsclm.prs_cs_clump."""
import numpy as np
import pytest
from moirais.fn.prsclm import prs_cs_clump


def test_prsclm_basic():
    """Test basic functionality."""
    sumstats = np.random.default_rng(42).normal(0, 1, 100)
    ld_ref = np.random.default_rng(42).normal(0, 1, 100)
    p_threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = prs_cs_clump(sumstats, ld_ref, p_threshold)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prsclm_edge():
    """Test edge cases."""
    sumstats = np.random.default_rng(42).normal(0, 1, 100)
    ld_ref = np.random.default_rng(42).normal(0, 1, 100)
    p_threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = prs_cs_clump(sumstats, ld_ref, p_threshold)
    assert isinstance(result, dict)
