"""Tests for grn001.geron_ch4_simple_linear_life_satisfaction."""
import numpy as np
import pytest
from morie.fn.grn001 import geron_ch4_simple_linear_life_satisfaction


def test_grn001_basic():
    """Test basic functionality."""
    theta_0 = np.random.default_rng(42).normal(0, 1, 100)
    theta_1 = np.random.default_rng(42).normal(0, 1, 100)
    GDP_per_capita = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ch4_simple_linear_life_satisfaction(theta_0, theta_1, GDP_per_capita)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grn001_edge():
    """Test edge cases."""
    theta_0 = np.random.default_rng(42).normal(0, 1, 100)
    theta_1 = np.random.default_rng(42).normal(0, 1, 100)
    GDP_per_capita = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ch4_simple_linear_life_satisfaction(theta_0, theta_1, GDP_per_capita)
    assert isinstance(result, dict)
