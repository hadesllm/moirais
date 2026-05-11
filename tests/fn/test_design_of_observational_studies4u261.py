"""Tests for design_of_observational_studies4u261.design_of_observational_studies_chapter_4_unnumbered_261."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies4u261 import design_of_observational_studies_chapter_4_unnumbered_261


def test_design_of_observational_studies4u261_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_4_unnumbered_261(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies4u261_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_4_unnumbered_261(x)
    assert isinstance(result, dict)
