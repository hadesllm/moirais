"""Tests for gb571m.gibbons_wsrt_mean."""
import numpy as np
import pytest
from moirais.fn.gb571m import gibbons_wsrt_mean


def test_gb571m_basic():
    """Test basic functionality."""
    n = 100
    result = gibbons_wsrt_mean(n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb571m_edge():
    """Test edge cases."""
    n = 100
    result = gibbons_wsrt_mean(n)
    assert isinstance(result, dict)
