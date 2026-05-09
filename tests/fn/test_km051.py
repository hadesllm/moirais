"""Tests for km051.kamath_ch3_qa_trigger_template."""
import numpy as np
import pytest
from moirais.fn.km051 import kamath_ch3_qa_trigger_template


def test_km051_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    z_adv = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch3_qa_trigger_template(x, y, T, z_adv)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km051_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    z_adv = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch3_qa_trigger_template(x, y, T, z_adv)
    assert isinstance(result, dict)
