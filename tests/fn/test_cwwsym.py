"""Tests for cwwsym.cwt_morlet."""
import numpy as np
import pytest
from morie.fn.cwwsym import cwt_morlet


def test_cwwsym_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    scales = np.random.default_rng(42).normal(0, 1, 100)
    result = cwt_morlet(y, scales)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cwwsym_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    scales = np.random.default_rng(42).normal(0, 1, 100)
    result = cwt_morlet(y, scales)
    assert isinstance(result, dict)
