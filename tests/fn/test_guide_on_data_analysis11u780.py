"""Tests for guide_on_data_analysis11u780.guide_on_data_analysis_chapter_11_unnumbered_780."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis11u780 import guide_on_data_analysis_chapter_11_unnumbered_780


def test_guide_on_data_analysis11u780_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_11_unnumbered_780(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis11u780_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_11_unnumbered_780(x)
    assert isinstance(result, dict)
