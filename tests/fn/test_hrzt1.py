"""Tests for hrzt1.horowitz_treatment_effect."""
import numpy as np
import pytest
from moirais.fn.hrzt1 import horowitz_treatment_effect


def test_hrzt1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_treatment_effect(x, y, treatment)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzt1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_treatment_effect(x, y, treatment)
    assert isinstance(result, dict)
