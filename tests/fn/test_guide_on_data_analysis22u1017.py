"""Tests for guide_on_data_analysis22u1017.guide_on_data_analysis_chapter_22_unnumbered_1017."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis22u1017 import guide_on_data_analysis_chapter_22_unnumbered_1017


def test_guide_on_data_analysis22u1017_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_22_unnumbered_1017(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis22u1017_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_22_unnumbered_1017(x)
    assert isinstance(result, dict)
