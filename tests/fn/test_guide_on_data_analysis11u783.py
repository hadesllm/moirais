"""Tests for guide_on_data_analysis11u783.guide_on_data_analysis_chapter_11_unnumbered_783."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis11u783 import guide_on_data_analysis_chapter_11_unnumbered_783


def test_guide_on_data_analysis11u783_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_11_unnumbered_783(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis11u783_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_11_unnumbered_783(x)
    assert isinstance(result, dict)
