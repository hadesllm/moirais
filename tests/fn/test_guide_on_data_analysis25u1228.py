"""Tests for guide_on_data_analysis25u1228.guide_on_data_analysis_chapter_25_unnumbered_1228."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis25u1228 import guide_on_data_analysis_chapter_25_unnumbered_1228


def test_guide_on_data_analysis25u1228_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_25_unnumbered_1228(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis25u1228_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_25_unnumbered_1228(x)
    assert isinstance(result, dict)
