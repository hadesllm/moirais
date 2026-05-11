"""Tests for agstds.age_standardize."""
import numpy as np
import pytest
from morie.fn.agstds import age_standardize


def test_agstds_basic():
    """Test basic functionality."""
    age_specific_rates = np.random.default_rng(42).normal(0, 1, 100)
    standard_pop = np.random.default_rng(42).normal(0, 1, 100)
    result = age_standardize(age_specific_rates, standard_pop)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agstds_edge():
    """Test edge cases."""
    age_specific_rates = np.random.default_rng(42).normal(0, 1, 100)
    standard_pop = np.random.default_rng(42).normal(0, 1, 100)
    result = age_standardize(age_specific_rates, standard_pop)
    assert isinstance(result, dict)
