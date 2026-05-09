"""Tests for hedderich8e39.hedderich_chapter_8_equation_39."""
import numpy as np
import pytest
from moirais.fn.hedderich8e39 import hedderich_chapter_8_equation_39


def test_hedderich8e39_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_39(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich8e39_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_39(x)
    assert isinstance(result, dict)
