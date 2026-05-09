"""Tests for spsann.schabenberger_simulated_annealing."""
import numpy as np
import pytest
from moirais.fn.spsann import schabenberger_simulated_annealing


def test_spsann_basic():
    """Test basic functionality."""
    target_stats = np.random.default_rng(42).normal(0, 1, 100)
    initial_config = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_simulated_annealing(target_stats, initial_config)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spsann_edge():
    """Test edge cases."""
    target_stats = np.random.default_rng(42).normal(0, 1, 100)
    initial_config = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_simulated_annealing(target_stats, initial_config)
    assert isinstance(result, dict)
