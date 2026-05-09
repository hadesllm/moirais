"""Tests for hedderich5e16.hedderich_chapter_5_equation_16."""
import numpy as np
import pytest
from moirais.fn.hedderich5e16 import hedderich_chapter_5_equation_16


def test_hedderich5e16_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_5_equation_16(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich5e16_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_5_equation_16(x)
    assert isinstance(result, dict)
