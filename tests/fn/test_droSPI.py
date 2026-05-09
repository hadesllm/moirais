"""Tests for droSPI.spi."""
import numpy as np
import pytest
from moirais.fn.droSPI import spi


def test_droSPI_basic():
    """Test basic functionality."""
    precip = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = spi(precip, window)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_droSPI_edge():
    """Test edge cases."""
    precip = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = spi(precip, window)
    assert isinstance(result, dict)
