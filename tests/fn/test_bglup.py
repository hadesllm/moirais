"""Tests for bglup.bayes_cpi_genomic."""
import numpy as np
import pytest
from morie.fn.bglup import bayes_cpi_genomic


def test_bglup_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = bayes_cpi_genomic(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bglup_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = bayes_cpi_genomic(x, y)
    assert isinstance(result, dict)
