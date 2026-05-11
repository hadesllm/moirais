"""Tests for bootss.bootstrap_survey."""
import numpy as np
import pytest
from morie.fn.bootss import bootstrap_survey


def test_bootss_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    strata = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = bootstrap_survey(data, strata, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bootss_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    strata = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = bootstrap_survey(data, strata, B)
    assert isinstance(result, dict)
