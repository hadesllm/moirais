"""Tests for gb921m.gibbons_mood_moments."""
import numpy as np
import pytest
from morie.fn.gb921m import gibbons_mood_moments


def test_gb921m_basic():
    """Test basic functionality."""
    m = 10
    n = 100
    N = 100
    result = gibbons_mood_moments(m, n, N)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb921m_edge():
    """Test edge cases."""
    m = 10
    n = 100
    N = 100
    result = gibbons_mood_moments(m, n, N)
    assert isinstance(result, dict)
