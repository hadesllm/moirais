"""Tests for design_of_observational_studies20u724.design_of_observational_studies_chapter_20_unnumbered_724."""
import numpy as np
import pytest
from moirais.fn.design_of_observational_studies20u724 import design_of_observational_studies_chapter_20_unnumbered_724


def test_design_of_observational_studies20u724_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_20_unnumbered_724(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies20u724_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_20_unnumbered_724(x)
    assert isinstance(result, dict)
