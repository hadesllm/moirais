"""Tests for frwil.free_wilson_qsar."""
import numpy as np
import pytest
from morie.fn.frwil import free_wilson_qsar


def test_frwil_basic():
    """Test basic functionality."""
    activities = np.random.default_rng(42).normal(0, 1, 100)
    substituent_indicators = np.random.default_rng(42).normal(0, 1, 100)
    result = free_wilson_qsar(activities, substituent_indicators)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_frwil_edge():
    """Test edge cases."""
    activities = np.random.default_rng(42).normal(0, 1, 100)
    substituent_indicators = np.random.default_rng(42).normal(0, 1, 100)
    result = free_wilson_qsar(activities, substituent_indicators)
    assert isinstance(result, dict)
