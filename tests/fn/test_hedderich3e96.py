"""Tests for hedderich3e96.hedderich_chapter_3_equation_96."""
import numpy as np
import pytest
from morie.fn.hedderich3e96 import hedderich_chapter_3_equation_96


def test_hedderich3e96_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_3_equation_96(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich3e96_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_3_equation_96(x)
    assert isinstance(result, dict)
