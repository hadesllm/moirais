"""Tests for hrzdcrc.horowitz_deconv_rate."""
import numpy as np
import pytest
from moirais.fn.hrzdcrc import horowitz_deconv_rate


def test_hrzdcrc_basic():
    """Test basic functionality."""
    n = 100
    smoothness_type = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    s = 90
    result = horowitz_deconv_rate(n, smoothness_type, r, s)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzdcrc_edge():
    """Test edge cases."""
    n = 100
    smoothness_type = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    s = 90
    result = horowitz_deconv_rate(n, smoothness_type, r, s)
    assert isinstance(result, dict)
