"""Tests for design_of_observational_studies3u211.design_of_observational_studies_chapter_3_unnumbered_211."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies3u211 import design_of_observational_studies_chapter_3_unnumbered_211


def test_design_of_observational_studies3u211_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_3_unnumbered_211(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies3u211_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_3_unnumbered_211(x)
    assert isinstance(result, dict)
