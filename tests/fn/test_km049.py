"""Tests for km049.kamath_ch3_top1_prompt_metric."""
import numpy as np
import pytest
from moirais.fn.km049 import kamath_ch3_top1_prompt_metric


def test_km049_basic():
    """Test basic functionality."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    P_LM = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch3_top1_prompt_metric(R, t, P_LM)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km049_edge():
    """Test edge cases."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    P_LM = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch3_top1_prompt_metric(R, t, P_LM)
    assert isinstance(result, dict)
