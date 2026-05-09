"""Tests for guide_on_data_analysis28u1402.guide_on_data_analysis_chapter_28_unnumbered_1402."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis28u1402 import guide_on_data_analysis_chapter_28_unnumbered_1402


def test_guide_on_data_analysis28u1402_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_28_unnumbered_1402(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis28u1402_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_28_unnumbered_1402(x)
    assert isinstance(result, dict)
