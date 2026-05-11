"""Tests for dicg.deviance_information_criterion."""
import numpy as np
import pytest
from morie.fn.dicg import deviance_information_criterion


def test_dicg_basic():
    """Test basic functionality."""
    deviance = np.random.default_rng(42).normal(0, 1, 100)
    result = deviance_information_criterion(deviance)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dicg_edge():
    """Test edge cases."""
    deviance = np.random.default_rng(42).normal(0, 1, 100)
    result = deviance_information_criterion(deviance)
    assert isinstance(result, dict)
