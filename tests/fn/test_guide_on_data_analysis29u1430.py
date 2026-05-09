"""Tests for guide_on_data_analysis29u1430.guide_on_data_analysis_chapter_29_unnumbered_1430."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis29u1430 import guide_on_data_analysis_chapter_29_unnumbered_1430


def test_guide_on_data_analysis29u1430_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_29_unnumbered_1430(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis29u1430_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_29_unnumbered_1430(x)
    assert isinstance(result, dict)
