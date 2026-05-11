"""Tests for alsft.alammar_setfit_twostep."""
import numpy as np
import pytest
from morie.fn.alsft import alammar_setfit_twostep


def test_alsft_basic():
    """Test basic functionality."""
    few_shot_pairs = np.random.default_rng(42).normal(0, 1, 100)
    encoder = np.random.default_rng(42).normal(0, 1, 100)
    classifier = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_setfit_twostep(few_shot_pairs, encoder, classifier)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alsft_edge():
    """Test edge cases."""
    few_shot_pairs = np.random.default_rng(42).normal(0, 1, 100)
    encoder = np.random.default_rng(42).normal(0, 1, 100)
    classifier = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_setfit_twostep(few_shot_pairs, encoder, classifier)
    assert isinstance(result, dict)
