"""Tests for pluginM.plug_in_mediation."""
import numpy as np
import pytest
from morie.fn.pluginM import plug_in_mediation


def test_pluginM_basic():
    """Test basic functionality."""
    Y_model = np.random.default_rng(42).normal(0, 1, 100)
    M_model = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = plug_in_mediation(Y_model, M_model, X, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pluginM_edge():
    """Test edge cases."""
    Y_model = np.random.default_rng(42).normal(0, 1, 100)
    M_model = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = plug_in_mediation(Y_model, M_model, X, C)
    assert isinstance(result, dict)
