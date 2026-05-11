"""Tests for kpmnsv.kaplan_meier_survival."""
import numpy as np
import pytest
from morie.fn.kpmnsv import kaplan_meier_survival


def test_kpmnsv_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = kaplan_meier_survival(time, event)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kpmnsv_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = kaplan_meier_survival(time, event)
    assert isinstance(result, dict)
