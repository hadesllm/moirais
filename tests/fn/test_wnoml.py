"""Tests for wnoml.wnominate_logit."""
import numpy as np
import pytest
from morie.fn.wnoml import wnominate_logit


def test_wnoml_basic():
    """Test basic functionality."""
    votes = np.random.default_rng(43).integers(0, 2, (50, 100))
    ideal_points = np.random.default_rng(42).normal(0, 1, 100)
    yea_nay_positions = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = wnominate_logit(votes, ideal_points, yea_nay_positions, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wnoml_edge():
    """Test edge cases."""
    votes = np.random.default_rng(43).integers(0, 2, (50, 100))
    ideal_points = np.random.default_rng(42).normal(0, 1, 100)
    yea_nay_positions = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = wnominate_logit(votes, ideal_points, yea_nay_positions, beta)
    assert isinstance(result, dict)
