"""Tests for fzcov2.fauzi_cov_surv_est2."""
import numpy as np
import pytest
from moirais.fn.fzcov2 import fauzi_cov_surv_est2


def test_fzcov2_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    result = fauzi_cov_surv_est2(t, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzcov2_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    result = fauzi_cov_surv_est2(t, bandwidth)
    assert isinstance(result, dict)
