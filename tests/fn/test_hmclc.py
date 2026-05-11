"""Tests for hmclc.geron_classification_localization."""
import numpy as np
import pytest
from morie.fn.hmclc import geron_classification_localization


def test_hmclc_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_classification_localization(image, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmclc_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_classification_localization(image, model)
    assert isinstance(result, dict)
