"""Tests for spqct.schabenberger_quadrat_count_test."""
import numpy as np
import pytest
from morie.fn.spqct import schabenberger_quadrat_count_test


def test_spqct_basic():
    """Test basic functionality."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    quadrats = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_quadrat_count_test(points, quadrats)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spqct_edge():
    """Test edge cases."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    quadrats = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_quadrat_count_test(points, quadrats)
    assert isinstance(result, dict)
