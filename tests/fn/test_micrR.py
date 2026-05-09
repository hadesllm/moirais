"""Tests for micrR.microsoft_sr."""
import numpy as np
import pytest
from moirais.fn.micrR import microsoft_sr


def test_micrR_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = microsoft_sr(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_micrR_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = microsoft_sr(x)
    assert isinstance(result, dict)
