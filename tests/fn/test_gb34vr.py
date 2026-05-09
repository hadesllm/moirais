"""Tests for gb34vr.gibbons_runs_ud_var."""
import numpy as np
import pytest
from moirais.fn.gb34vr import gibbons_runs_ud_var


def test_gb34vr_basic():
    """Test basic functionality."""
    n = 100
    result = gibbons_runs_ud_var(n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb34vr_edge():
    """Test edge cases."""
    n = 100
    result = gibbons_runs_ud_var(n)
    assert isinstance(result, dict)
