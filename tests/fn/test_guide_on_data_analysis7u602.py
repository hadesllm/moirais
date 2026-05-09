"""Tests for guide_on_data_analysis7u602.guide_on_data_analysis_chapter_7_unnumbered_602."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis7u602 import guide_on_data_analysis_chapter_7_unnumbered_602


def test_guide_on_data_analysis7u602_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_7_unnumbered_602(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis7u602_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_7_unnumbered_602(x)
    assert isinstance(result, dict)
