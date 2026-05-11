"""Tests for alaug.alammar_augmented_sbert."""
import numpy as np
import pytest
from morie.fn.alaug import alammar_augmented_sbert


def test_alaug_basic():
    """Test basic functionality."""
    unlabeled_pairs = np.random.default_rng(42).normal(0, 1, 100)
    cross_encoder = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_augmented_sbert(unlabeled_pairs, cross_encoder)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alaug_edge():
    """Test edge cases."""
    unlabeled_pairs = np.random.default_rng(42).normal(0, 1, 100)
    cross_encoder = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_augmented_sbert(unlabeled_pairs, cross_encoder)
    assert isinstance(result, dict)
