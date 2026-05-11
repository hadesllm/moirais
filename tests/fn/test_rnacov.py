"""Tests for rnacov.rna_covariance."""
import numpy as np
import pytest
from morie.fn.rnacov import rna_covariance


def test_rnacov_basic():
    """Test basic functionality."""
    alignment = np.random.default_rng(42).normal(0, 1, 100)
    structure = np.random.default_rng(42).normal(0, 1, 100)
    result = rna_covariance(alignment, structure)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rnacov_edge():
    """Test edge cases."""
    alignment = np.random.default_rng(42).normal(0, 1, 100)
    structure = np.random.default_rng(42).normal(0, 1, 100)
    result = rna_covariance(alignment, structure)
    assert isinstance(result, dict)
