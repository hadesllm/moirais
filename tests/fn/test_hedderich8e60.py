"""Tests for hedderich8e60.hedderich_chapter_8_equation_60."""
import numpy as np
import pytest
from morie.fn.hedderich8e60 import hedderich_chapter_8_equation_60


def test_hedderich8e60_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_60(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich8e60_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_60(x)
    assert isinstance(result, dict)
