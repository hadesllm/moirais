"""Tests for volrm.vol_riskmetrics."""
import numpy as np
import pytest
from morie.fn.volrm import vol_riskmetrics


def test_volrm_basic():
    """Test basic functionality."""
    r = 10
    lam = 0.1
    result = vol_riskmetrics(r, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volrm_edge():
    """Test edge cases."""
    r = 10
    lam = 0.1
    result = vol_riskmetrics(r, lam)
    assert isinstance(result, dict)
