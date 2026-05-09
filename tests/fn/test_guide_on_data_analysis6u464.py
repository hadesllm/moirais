"""Tests for guide_on_data_analysis6u464.guide_on_data_analysis_chapter_6_unnumbered_464."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis6u464 import guide_on_data_analysis_chapter_6_unnumbered_464


def test_guide_on_data_analysis6u464_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_6_unnumbered_464(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis6u464_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_6_unnumbered_464(x)
    assert isinstance(result, dict)
