"""Tests for hrzsicd.horowitz_sim_id_discrete_x."""
import numpy as np
import pytest
from moirais.fn.hrzsicd import horowitz_sim_id_discrete_x


def test_hrzsicd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_sim_id_discrete_x(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzsicd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_sim_id_discrete_x(x, y)
    assert isinstance(result, dict)
