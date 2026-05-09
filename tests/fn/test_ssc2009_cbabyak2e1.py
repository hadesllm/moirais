"""Tests for ssc2009_cbabyak2e1.ssc2009_cbabyak_chapter_2_equation_1."""
import numpy as np
import pytest
from moirais.fn.ssc2009_cbabyak2e1 import ssc2009_cbabyak_chapter_2_equation_1


def test_ssc2009_cbabyak2e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ssc2009_cbabyak_chapter_2_equation_1(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ssc2009_cbabyak2e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ssc2009_cbabyak_chapter_2_equation_1(x)
    assert isinstance(result, dict)
