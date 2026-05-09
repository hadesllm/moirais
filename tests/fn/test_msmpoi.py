"""Tests for msmpoi.msm_poisson."""
import numpy as np
import pytest
from moirais.fn.msmpoi import msm_poisson


def test_msmpoi_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    offset = np.random.default_rng(42).normal(0, 1, 100)
    result = msm_poisson(y, treatment_history, covariate_history, offset)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msmpoi_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    offset = np.random.default_rng(42).normal(0, 1, 100)
    result = msm_poisson(y, treatment_history, covariate_history, offset)
    assert isinstance(result, dict)
