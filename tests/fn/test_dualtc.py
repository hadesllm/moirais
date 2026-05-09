"""Tests for dualtc.dual_total_correlation."""
import numpy as np
import pytest
from moirais.fn.dualtc import dual_total_correlation


def test_dualtc_basic():
    """Test basic functionality."""
    p = 5
    result = dual_total_correlation(p)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_dualtc_edge():
    """Test edge cases."""
    p = 5
    result = dual_total_correlation(p)
    assert isinstance(result, dict)
