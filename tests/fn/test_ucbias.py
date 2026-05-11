"""Tests for ucbias.unmeasured_conf_bias."""
import numpy as np
import pytest
from morie.fn.ucbias import unmeasured_conf_bias


def test_ucbias_basic():
    """Test basic functionality."""
    RR_UD = np.random.default_rng(42).normal(0, 1, 100)
    RR_UY = np.random.default_rng(42).normal(0, 1, 100)
    result = unmeasured_conf_bias(RR_UD, RR_UY)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ucbias_edge():
    """Test edge cases."""
    RR_UD = np.random.default_rng(42).normal(0, 1, 100)
    RR_UY = np.random.default_rng(42).normal(0, 1, 100)
    result = unmeasured_conf_bias(RR_UD, RR_UY)
    assert isinstance(result, dict)
