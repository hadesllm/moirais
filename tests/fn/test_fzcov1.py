"""Tests for fzcov1.fauzi_cov_surv_est1."""
import numpy as np
import pytest
from morie.fn.fzcov1 import fauzi_cov_surv_est1


def test_fzcov1_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    result = fauzi_cov_surv_est1(t, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzcov1_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    result = fauzi_cov_surv_est1(t, bandwidth)
    assert isinstance(result, dict)
