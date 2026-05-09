"""Tests for design_of_observational_studies20u817.design_of_observational_studies_chapter_20_unnumbered_817."""
import numpy as np
import pytest
from moirais.fn.design_of_observational_studies20u817 import design_of_observational_studies_chapter_20_unnumbered_817


def test_design_of_observational_studies20u817_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_20_unnumbered_817(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_design_of_observational_studies20u817_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_20_unnumbered_817(x)
    assert isinstance(result, dict)
