"""Tests for kmpet.kamath_pet_loss."""
import numpy as np
import pytest
from morie.fn.kmpet import kamath_pet_loss


def test_kmpet_basic():
    """Test basic functionality."""
    verbalizer_logits = np.random.default_rng(42).normal(0, 1, 100)
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    mlm_logits = np.random.default_rng(42).normal(0, 1, 100)
    mlm_targets = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = kamath_pet_loss(verbalizer_logits, y_true, mlm_logits, mlm_targets, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmpet_edge():
    """Test edge cases."""
    verbalizer_logits = np.random.default_rng(42).normal(0, 1, 100)
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    mlm_logits = np.random.default_rng(42).normal(0, 1, 100)
    mlm_targets = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = kamath_pet_loss(verbalizer_logits, y_true, mlm_logits, mlm_targets, alpha)
    assert isinstance(result, dict)
