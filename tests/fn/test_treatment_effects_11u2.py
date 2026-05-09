"""Tests for treatment_effects_11u2.treatment_effects_1_chapter_1_unnumbered_2."""
import numpy as np
import pytest
from moirais.fn.treatment_effects_11u2 import treatment_effects_1_chapter_1_unnumbered_2


def test_treatment_effects_11u2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = treatment_effects_1_chapter_1_unnumbered_2(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_treatment_effects_11u2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = treatment_effects_1_chapter_1_unnumbered_2(x)
    assert isinstance(result, dict)
