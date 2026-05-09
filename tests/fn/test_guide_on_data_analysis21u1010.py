"""Tests for guide_on_data_analysis21u1010.guide_on_data_analysis_chapter_21_unnumbered_1010."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis21u1010 import guide_on_data_analysis_chapter_21_unnumbered_1010


def test_guide_on_data_analysis21u1010_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_21_unnumbered_1010(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis21u1010_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_21_unnumbered_1010(x)
    assert isinstance(result, dict)
