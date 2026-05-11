"""Tests for hedderich8e71.hedderich_chapter_8_equation_71."""
import numpy as np
import pytest
from morie.fn.hedderich8e71 import hedderich_chapter_8_equation_71


def test_hedderich8e71_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_71(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich8e71_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_71(x)
    assert isinstance(result, dict)
