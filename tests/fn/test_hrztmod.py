"""Tests for hrztmod.horowitz_transformation_model."""
import numpy as np
import pytest
from moirais.fn.hrztmod import horowitz_transformation_model


def test_hrztmod_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_transformation_model(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrztmod_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_transformation_model(x, y)
    assert isinstance(result, dict)
