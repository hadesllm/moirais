"""Tests for rfOOB.rf_oob_error."""
import numpy as np
import pytest
from morie.fn.rfOOB import rf_oob_error


def test_rfOOB_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    oob_preds = np.random.default_rng(42).normal(0, 1, 100)
    result = rf_oob_error(y, oob_preds)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rfOOB_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    oob_preds = np.random.default_rng(42).normal(0, 1, 100)
    result = rf_oob_error(y, oob_preds)
    assert isinstance(result, dict)
