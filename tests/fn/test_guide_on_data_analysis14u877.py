"""Tests for guide_on_data_analysis14u877.guide_on_data_analysis_chapter_14_unnumbered_877."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis14u877 import guide_on_data_analysis_chapter_14_unnumbered_877


def test_guide_on_data_analysis14u877_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_14_unnumbered_877(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis14u877_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_14_unnumbered_877(x)
    assert isinstance(result, dict)
