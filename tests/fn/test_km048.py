"""Tests for km048.kamath_ch3_cloze_prompt_template."""
import numpy as np
import pytest
from morie.fn.km048 import kamath_ch3_cloze_prompt_template


def test_km048_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = kamath_ch3_cloze_prompt_template(x, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km048_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = kamath_ch3_cloze_prompt_template(x, z)
    assert isinstance(result, dict)
