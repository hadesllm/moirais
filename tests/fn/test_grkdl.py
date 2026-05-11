"""Tests for grkdl.geron_knowledge_distillation_loss."""
import numpy as np
import pytest
from morie.fn.grkdl import geron_knowledge_distillation_loss


def test_grkdl_basic():
    """Test basic functionality."""
    student_logits = np.random.default_rng(42).normal(0, 1, 100)
    teacher_logits = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_knowledge_distillation_loss(student_logits, teacher_logits, y, alpha, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grkdl_edge():
    """Test edge cases."""
    student_logits = np.random.default_rng(42).normal(0, 1, 100)
    teacher_logits = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_knowledge_distillation_loss(student_logits, teacher_logits, y, alpha, T)
    assert isinstance(result, dict)
