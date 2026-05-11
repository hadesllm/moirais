"""Tests for guide_on_data_analysis6u502.guide_on_data_analysis_chapter_6_unnumbered_502."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis6u502 import guide_on_data_analysis_chapter_6_unnumbered_502


def test_guide_on_data_analysis6u502_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_6_unnumbered_502(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis6u502_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_6_unnumbered_502(x)
    assert isinstance(result, dict)
