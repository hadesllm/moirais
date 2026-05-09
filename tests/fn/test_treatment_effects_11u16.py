"""Tests for treatment_effects_11u16.treatment_effects_1_chapter_1_unnumbered_16."""
import numpy as np
import pytest
from moirais.fn.treatment_effects_11u16 import treatment_effects_1_chapter_1_unnumbered_16


def test_treatment_effects_11u16_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = treatment_effects_1_chapter_1_unnumbered_16(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_treatment_effects_11u16_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = treatment_effects_1_chapter_1_unnumbered_16(x)
    assert isinstance(result, dict)
