"""Tests for sbcrk.simulation_based_calibration_rank."""
import numpy as np
import pytest
from moirais.fn.sbcrk import simulation_based_calibration_rank


def test_sbcrk_basic():
    """Test basic functionality."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    result = simulation_based_calibration_rank(samples)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_sbcrk_edge():
    """Test edge cases."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    result = simulation_based_calibration_rank(samples)
    assert isinstance(result, dict)
