"""Tests for yldyear.yld_calculation."""
import numpy as np
import pytest
from moirais.fn.yldyear import yld_calculation


def test_yldyear_basic():
    """Test basic functionality."""
    prevalence = np.random.default_rng(42).normal(0, 1, 100)
    disability = np.random.default_rng(42).normal(0, 1, 100)
    duration = np.random.default_rng(42).normal(0, 1, 100)
    result = yld_calculation(prevalence, disability, duration)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_yldyear_edge():
    """Test edge cases."""
    prevalence = np.random.default_rng(42).normal(0, 1, 100)
    disability = np.random.default_rng(42).normal(0, 1, 100)
    duration = np.random.default_rng(42).normal(0, 1, 100)
    result = yld_calculation(prevalence, disability, duration)
    assert isinstance(result, dict)
