"""Tests for survipa.ipa_brier."""
import numpy as np
import pytest
from morie.fn.survipa import ipa_brier


def test_survipa_basic():
    """Test basic functionality."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    null_fit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = ipa_brier(fit, null_fit, time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survipa_edge():
    """Test edge cases."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    null_fit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = ipa_brier(fit, null_fit, time)
    assert isinstance(result, dict)
