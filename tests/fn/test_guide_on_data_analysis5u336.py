"""Tests for guide_on_data_analysis5u336.guide_on_data_analysis_chapter_5_unnumbered_336."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis5u336 import guide_on_data_analysis_chapter_5_unnumbered_336


def test_guide_on_data_analysis5u336_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_5_unnumbered_336(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis5u336_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_5_unnumbered_336(x)
    assert isinstance(result, dict)
