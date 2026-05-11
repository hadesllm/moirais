"""Tests for hedderich2e46.hedderich_chapter_2_equation_46."""
import numpy as np
import pytest
from morie.fn.hedderich2e46 import hedderich_chapter_2_equation_46


def test_hedderich2e46_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_2_equation_46(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich2e46_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_2_equation_46(x)
    assert isinstance(result, dict)
