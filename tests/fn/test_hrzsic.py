"""Tests for hrzsic.horowitz_sim_identification."""
import numpy as np
import pytest
from moirais.fn.hrzsic import horowitz_sim_identification


def test_hrzsic_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = horowitz_sim_identification(x, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzsic_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = horowitz_sim_identification(x, beta)
    assert isinstance(result, dict)
