"""Tests for guide_on_data_analysis22u1020.guide_on_data_analysis_chapter_22_unnumbered_1020."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis22u1020 import guide_on_data_analysis_chapter_22_unnumbered_1020


def test_guide_on_data_analysis22u1020_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_22_unnumbered_1020(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis22u1020_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_22_unnumbered_1020(x)
    assert isinstance(result, dict)
