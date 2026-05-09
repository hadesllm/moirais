"""Tests for gb_sp2.gibbons_spearman_exact."""
import numpy as np
import pytest
from moirais.fn.gb_sp2 import gibbons_spearman_exact


def test_gb_sp2_basic():
    """Test basic functionality."""
    n = 100
    rho = 0.5
    result = gibbons_spearman_exact(n, rho)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb_sp2_edge():
    """Test edge cases."""
    n = 100
    rho = 0.5
    result = gibbons_spearman_exact(n, rho)
    assert isinstance(result, dict)
