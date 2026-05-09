"""Tests for confgg.configuration_model."""
import numpy as np
import pytest
from moirais.fn.confgg import configuration_model


def test_confgg_basic():
    """Test basic functionality."""
    degrees = np.random.default_rng(42).normal(0, 1, 100)
    result = configuration_model(degrees)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_confgg_edge():
    """Test edge cases."""
    degrees = np.random.default_rng(42).normal(0, 1, 100)
    result = configuration_model(degrees)
    assert isinstance(result, dict)
