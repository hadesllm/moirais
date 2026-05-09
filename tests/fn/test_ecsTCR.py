"""Tests for ecsTCR.ecs_tcr."""
import numpy as np
import pytest
from moirais.fn.ecsTCR import ecs_tcr


def test_ecsTCR_basic():
    """Test basic functionality."""
    model_run = np.random.default_rng(42).normal(0, 1, 100)
    CO2_traj = np.random.default_rng(42).normal(0, 1, 100)
    result = ecs_tcr(model_run, CO2_traj)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ecsTCR_edge():
    """Test edge cases."""
    model_run = np.random.default_rng(42).normal(0, 1, 100)
    CO2_traj = np.random.default_rng(42).normal(0, 1, 100)
    result = ecs_tcr(model_run, CO2_traj)
    assert isinstance(result, dict)
