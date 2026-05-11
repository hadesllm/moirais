"""Tests for gb571.gibbons_wilcoxon_signed_rank."""
import numpy as np
import pytest
from morie.fn.gb571 import gibbons_wilcoxon_signed_rank


def test_gb571_basic():
    """Test basic functionality."""
    differences = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_wilcoxon_signed_rank(differences)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb571_edge():
    """Test edge cases."""
    differences = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_wilcoxon_signed_rank(differences)
    assert isinstance(result, dict)
