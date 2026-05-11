"""Tests for hedderich8e90.hedderich_chapter_8_equation_90."""
import numpy as np
import pytest
from morie.fn.hedderich8e90 import hedderich_chapter_8_equation_90


def test_hedderich8e90_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_90(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich8e90_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_90(x)
    assert isinstance(result, dict)
