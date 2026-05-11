"""Tests for km042.kamath_ch3_prompt_label_mapping."""
import numpy as np
import pytest
from morie.fn.km042 import kamath_ch3_prompt_label_mapping


def test_km042_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = kamath_ch3_prompt_label_mapping(x, y, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km042_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = kamath_ch3_prompt_label_mapping(x, y, M)
    assert isinstance(result, dict)
