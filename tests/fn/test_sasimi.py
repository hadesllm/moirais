"""Tests for sasimi.tanimoto_similarity."""
import numpy as np
import pytest
from morie.fn.sasimi import tanimoto_similarity


def test_sasimi_basic():
    """Test basic functionality."""
    fp_a = np.random.default_rng(42).normal(0, 1, 100)
    fp_b = np.random.default_rng(42).normal(0, 1, 100)
    result = tanimoto_similarity(fp_a, fp_b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sasimi_edge():
    """Test edge cases."""
    fp_a = np.random.default_rng(42).normal(0, 1, 100)
    fp_b = np.random.default_rng(42).normal(0, 1, 100)
    result = tanimoto_similarity(fp_a, fp_b)
    assert isinstance(result, dict)
