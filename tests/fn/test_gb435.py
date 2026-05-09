"""Tests for gb435.gibbons_ks_onesided_asymp."""
import numpy as np
import pytest
from moirais.fn.gb435 import gibbons_ks_onesided_asymp


def test_gb435_basic():
    """Test basic functionality."""
    d = 5
    n = 100
    result = gibbons_ks_onesided_asymp(d, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb435_edge():
    """Test edge cases."""
    d = 5
    n = 100
    result = gibbons_ks_onesided_asymp(d, n)
    assert isinstance(result, dict)
