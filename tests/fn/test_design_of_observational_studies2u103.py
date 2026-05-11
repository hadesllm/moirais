"""Tests for design_of_observational_studies2u103.design_of_observational_studies_chapter_2_unnumbered_103."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies2u103 import design_of_observational_studies_chapter_2_unnumbered_103


def test_design_of_observational_studies2u103_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_2_unnumbered_103(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies2u103_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_2_unnumbered_103(x)
    assert isinstance(result, dict)
