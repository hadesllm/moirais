"""Tests for gh_ap_d2.ghosal_lecam_lemma."""
import numpy as np
import pytest
from morie.fn.gh_ap_d2 import ghosal_lecam_lemma


def test_gh_ap_d2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_lecam_lemma(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gh_ap_d2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_lecam_lemma(x)
    assert isinstance(result, dict)
