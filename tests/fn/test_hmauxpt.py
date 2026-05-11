"""Tests for hmauxpt.geron_auxiliary_task_pretraining."""
import numpy as np
import pytest
from morie.fn.hmauxpt import geron_auxiliary_task_pretraining


def test_hmauxpt_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    aux_data = np.random.default_rng(42).normal(0, 1, 100)
    target_data = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_auxiliary_task_pretraining(model, aux_data, target_data)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmauxpt_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    aux_data = np.random.default_rng(42).normal(0, 1, 100)
    target_data = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_auxiliary_task_pretraining(model, aux_data, target_data)
    assert isinstance(result, dict)
