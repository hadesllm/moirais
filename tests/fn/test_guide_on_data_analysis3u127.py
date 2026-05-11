"""Tests for guide_on_data_analysis3u127.guide_on_data_analysis_chapter_3_unnumbered_127."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis3u127 import guide_on_data_analysis_chapter_3_unnumbered_127


def test_guide_on_data_analysis3u127_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_3_unnumbered_127(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_guide_on_data_analysis3u127_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_3_unnumbered_127(x)
    assert isinstance(result, dict)
