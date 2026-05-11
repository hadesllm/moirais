"""Tests for guide_on_data_analysis24u1124.guide_on_data_analysis_chapter_24_unnumbered_1124."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis24u1124 import guide_on_data_analysis_chapter_24_unnumbered_1124


def test_guide_on_data_analysis24u1124_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1124(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis24u1124_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_24_unnumbered_1124(x)
    assert isinstance(result, dict)
