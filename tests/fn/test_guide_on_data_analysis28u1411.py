"""Tests for guide_on_data_analysis28u1411.guide_on_data_analysis_chapter_28_unnumbered_1411."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis28u1411 import guide_on_data_analysis_chapter_28_unnumbered_1411


def test_guide_on_data_analysis28u1411_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_28_unnumbered_1411(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_guide_on_data_analysis28u1411_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_28_unnumbered_1411(x)
    assert isinstance(result, dict)
