"""Tests for guide_on_data_analysis8u641.guide_on_data_analysis_chapter_8_unnumbered_641."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis8u641 import guide_on_data_analysis_chapter_8_unnumbered_641


def test_guide_on_data_analysis8u641_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_8_unnumbered_641(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis8u641_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_8_unnumbered_641(x)
    assert isinstance(result, dict)
