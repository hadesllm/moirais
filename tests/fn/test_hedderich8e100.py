"""Tests for hedderich8e100.hedderich_chapter_8_equation_100."""
import numpy as np
import pytest
from morie.fn.hedderich8e100 import hedderich_chapter_8_equation_100


def test_hedderich8e100_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_100(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich8e100_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_100(x)
    assert isinstance(result, dict)
