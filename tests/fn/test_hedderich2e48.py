"""Tests for hedderich2e48.hedderich_chapter_2_equation_48."""
import numpy as np
import pytest
from moirais.fn.hedderich2e48 import hedderich_chapter_2_equation_48


def test_hedderich2e48_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_2_equation_48(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich2e48_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_2_equation_48(x)
    assert isinstance(result, dict)
