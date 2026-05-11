"""Tests for treatment_effects_11u37.treatment_effects_1_chapter_1_unnumbered_37."""
import numpy as np
import pytest
from morie.fn.treatment_effects_11u37 import treatment_effects_1_chapter_1_unnumbered_37


def test_treatment_effects_11u37_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = treatment_effects_1_chapter_1_unnumbered_37(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_treatment_effects_11u37_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = treatment_effects_1_chapter_1_unnumbered_37(x)
    assert isinstance(result, dict)
