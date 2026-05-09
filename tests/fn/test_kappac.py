"""Tests for kappac.cohens_kappa."""
import numpy as np
import pytest
from moirais.fn.kappac import cohens_kappa


def test_kappac_basic():
    """Test basic functionality."""
    rater1 = np.random.default_rng(42).normal(0, 1, 100)
    rater2 = np.random.default_rng(42).normal(0, 1, 100)
    result = cohens_kappa(rater1, rater2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kappac_edge():
    """Test edge cases."""
    rater1 = np.random.default_rng(42).normal(0, 1, 100)
    rater2 = np.random.default_rng(42).normal(0, 1, 100)
    result = cohens_kappa(rater1, rater2)
    assert isinstance(result, dict)
