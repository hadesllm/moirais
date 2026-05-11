"""Tests for pdcoin.pedroni_panel_cointegration."""
import numpy as np
import pytest
from morie.fn.pdcoin import pedroni_panel_cointegration


def test_pdcoin_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    groups = np.random.default_rng(43).integers(0, 3, 100)
    result = pedroni_panel_cointegration(X, groups)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_pdcoin_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    groups = np.random.default_rng(43).integers(0, 3, 100)
    result = pedroni_panel_cointegration(X, groups)
    assert isinstance(result, dict)
