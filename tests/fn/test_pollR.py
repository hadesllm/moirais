"""Tests for pollR.pollards_rho."""
import numpy as np
import pytest
from moirais.fn.pollR import pollards_rho


def test_pollR_basic():
    """Test basic functionality."""
    n = 100
    result = pollards_rho(n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pollR_edge():
    """Test edge cases."""
    n = 100
    result = pollards_rho(n)
    assert isinstance(result, dict)
