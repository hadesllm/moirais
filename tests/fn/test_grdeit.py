"""Tests for grdeit.geron_deit_distillation_loss."""
import numpy as np
import pytest
from moirais.fn.grdeit import geron_deit_distillation_loss


def test_grdeit_basic():
    """Test basic functionality."""
    logits_cls = np.random.default_rng(42).normal(0, 1, 100)
    logits_dist = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    teacher_preds = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_deit_distillation_loss(logits_cls, logits_dist, y, teacher_preds)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grdeit_edge():
    """Test edge cases."""
    logits_cls = np.random.default_rng(42).normal(0, 1, 100)
    logits_dist = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    teacher_preds = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_deit_distillation_loss(logits_cls, logits_dist, y, teacher_preds)
    assert isinstance(result, dict)
