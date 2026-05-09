"""Tests for gb7381.gibbons_cs_null_var."""
import numpy as np
import pytest
from moirais.fn.gb7381 import gibbons_cs_null_var


def test_gb7381_basic():
    """Test basic functionality."""
    J = 20
    lam = 0.1
    result = gibbons_cs_null_var(J, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb7381_edge():
    """Test edge cases."""
    J = 20
    lam = 0.1
    result = gibbons_cs_null_var(J, lam)
    assert isinstance(result, dict)
