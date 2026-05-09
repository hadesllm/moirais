"""Tests for guide_on_data_analysis8u615.guide_on_data_analysis_chapter_8_unnumbered_615."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis8u615 import guide_on_data_analysis_chapter_8_unnumbered_615


def test_guide_on_data_analysis8u615_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_8_unnumbered_615(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis8u615_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_8_unnumbered_615(x)
    assert isinstance(result, dict)
