"""Tests for totcorr.total_correlation."""
import numpy as np
import pytest
from morie.fn.totcorr import total_correlation


def test_totcorr_basic():
    """Test basic functionality."""
    p = 5
    result = total_correlation(p)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_totcorr_edge():
    """Test edge cases."""
    p = 5
    result = total_correlation(p)
    assert isinstance(result, dict)
