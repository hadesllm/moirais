"""Tests for guide_on_data_analysis27u1321.guide_on_data_analysis_chapter_27_unnumbered_1321."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis27u1321 import guide_on_data_analysis_chapter_27_unnumbered_1321


def test_guide_on_data_analysis27u1321_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_27_unnumbered_1321(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis27u1321_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_27_unnumbered_1321(x)
    assert isinstance(result, dict)
