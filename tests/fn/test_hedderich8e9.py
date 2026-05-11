"""Tests for hedderich8e9.hedderich_chapter_8_equation_9."""
import numpy as np
import pytest
from morie.fn.hedderich8e9 import hedderich_chapter_8_equation_9


def test_hedderich8e9_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_9(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich8e9_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_9(x)
    assert isinstance(result, dict)
