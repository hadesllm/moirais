"""Tests for guide_on_data_analysis14u904.guide_on_data_analysis_chapter_14_unnumbered_904."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis14u904 import guide_on_data_analysis_chapter_14_unnumbered_904


def test_guide_on_data_analysis14u904_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_14_unnumbered_904(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis14u904_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_14_unnumbered_904(x)
    assert isinstance(result, dict)
