"""Tests for kent_roach_benjamin_l_berger_emma_cunliffe_asad_g_kiyani_cri7u1.kent_roach_benjamin_l_berger_emma_cunliffe_asad_g_kiyani_cri_chapter_7_unnumbered_1."""
import numpy as np
import pytest
from morie.fn.kent_roach_benjamin_l_berger_emma_cunliffe_asad_g_kiyani_cri7u1 import kent_roach_benjamin_l_berger_emma_cunliffe_asad_g_kiyani_cri_chapter_7_unnumbered_1


def test_kent_roach_benjamin_l_berger_emma_cunliffe_asad_g_kiyani_cri7u1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kent_roach_benjamin_l_berger_emma_cunliffe_asad_g_kiyani_cri_chapter_7_unnumbered_1(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_kent_roach_benjamin_l_berger_emma_cunliffe_asad_g_kiyani_cri7u1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kent_roach_benjamin_l_berger_emma_cunliffe_asad_g_kiyani_cri_chapter_7_unnumbered_1(x)
    assert isinstance(result, dict)
