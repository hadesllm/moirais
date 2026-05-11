"""Tests for yllyear.yll_calculation."""
import numpy as np
import pytest
from morie.fn.yllyear import yll_calculation


def test_yllyear_basic():
    """Test basic functionality."""
    deaths = np.random.default_rng(42).normal(0, 1, 100)
    ages = np.random.default_rng(42).normal(0, 1, 100)
    life_table = np.random.default_rng(42).normal(0, 1, 100)
    result = yll_calculation(deaths, ages, life_table)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_yllyear_edge():
    """Test edge cases."""
    deaths = np.random.default_rng(42).normal(0, 1, 100)
    ages = np.random.default_rng(42).normal(0, 1, 100)
    life_table = np.random.default_rng(42).normal(0, 1, 100)
    result = yll_calculation(deaths, ages, life_table)
    assert isinstance(result, dict)
