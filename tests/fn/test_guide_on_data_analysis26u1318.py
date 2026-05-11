"""Tests for guide_on_data_analysis26u1318.guide_on_data_analysis_chapter_26_unnumbered_1318."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis26u1318 import guide_on_data_analysis_chapter_26_unnumbered_1318


def test_guide_on_data_analysis26u1318_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_26_unnumbered_1318(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis26u1318_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_26_unnumbered_1318(x)
    assert isinstance(result, dict)
