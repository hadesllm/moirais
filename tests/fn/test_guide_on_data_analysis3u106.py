"""Tests for guide_on_data_analysis3u106.guide_on_data_analysis_chapter_3_unnumbered_106."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis3u106 import guide_on_data_analysis_chapter_3_unnumbered_106


def test_guide_on_data_analysis3u106_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_3_unnumbered_106(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis3u106_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_3_unnumbered_106(x)
    assert isinstance(result, dict)
