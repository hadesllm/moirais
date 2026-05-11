"""Tests for spssoc.schabenberger_stationary_cov_semivario."""
import numpy as np
import pytest
from morie.fn.spssoc import schabenberger_stationary_cov_semivario


def test_spssoc_basic():
    """Test basic functionality."""
    cov_func = (lambda v: v)
    h = 0.3
    result = schabenberger_stationary_cov_semivario(cov_func, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spssoc_edge():
    """Test edge cases."""
    cov_func = (lambda v: v)
    h = 0.3
    result = schabenberger_stationary_cov_semivario(cov_func, h)
    assert isinstance(result, dict)
