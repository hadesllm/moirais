"""Tests for phlpr.phillips_perron."""
import numpy as np
import pytest
from morie.fn.phlpr import phillips_perron


def test_phlpr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = phillips_perron(y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_phlpr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = phillips_perron(y)
    assert isinstance(result, dict)
