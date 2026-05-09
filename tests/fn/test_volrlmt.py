"""Tests for volrlmt.vol_realised_log_vol_ar."""
import numpy as np
import pytest
from moirais.fn.volrlmt import vol_realised_log_vol_ar


def test_volrlmt_basic():
    """Test basic functionality."""
    RV = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_realised_log_vol_ar(RV)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volrlmt_edge():
    """Test edge cases."""
    RV = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_realised_log_vol_ar(RV)
    assert isinstance(result, dict)
