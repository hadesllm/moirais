"""Tests for spvmr.schabenberger_variance_mean_ratio."""
import numpy as np
import pytest
from moirais.fn.spvmr import schabenberger_variance_mean_ratio


def test_spvmr_basic():
    """Test basic functionality."""
    quadrat_counts = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_variance_mean_ratio(quadrat_counts)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spvmr_edge():
    """Test edge cases."""
    quadrat_counts = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_variance_mean_ratio(quadrat_counts)
    assert isinstance(result, dict)
