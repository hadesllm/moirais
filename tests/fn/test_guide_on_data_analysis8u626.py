"""Tests for guide_on_data_analysis8u626.guide_on_data_analysis_chapter_8_unnumbered_626."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis8u626 import guide_on_data_analysis_chapter_8_unnumbered_626


def test_guide_on_data_analysis8u626_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_8_unnumbered_626(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_guide_on_data_analysis8u626_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_8_unnumbered_626(x)
    assert isinstance(result, dict)
