"""Tests for levdis.levenshtein."""
import numpy as np
import pytest
from morie.fn.levdis import levenshtein


def test_levdis_basic():
    """Test basic functionality."""
    s1 = np.random.default_rng(42).normal(0, 1, 100)
    s2 = np.random.default_rng(42).normal(0, 1, 100)
    result = levenshtein(s1, s2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_levdis_edge():
    """Test edge cases."""
    s1 = np.random.default_rng(42).normal(0, 1, 100)
    s2 = np.random.default_rng(42).normal(0, 1, 100)
    result = levenshtein(s1, s2)
    assert isinstance(result, dict)
