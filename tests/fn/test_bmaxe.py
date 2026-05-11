"""Tests for morie.fn.bmaxe -- Bayesian model averaging."""

import numpy as np
from morie.fn.bmaxe import bayesian_model_averaging


def test_returns_dict():
    preds = np.array([[1, 2, 3], [4, 5, 6]])
    log_ml = np.array([-10.0, -5.0])
    result = bayesian_model_averaging(preds, log_ml)
    assert isinstance(result, dict)
    assert "averaged_predictions" in result


def test_weights_sum_to_one():
    preds = np.array([[1, 2], [3, 4], [5, 6]])
    log_ml = np.array([-10.0, -5.0, -8.0])
    result = bayesian_model_averaging(preds, log_ml)
    assert abs(sum(result["model_weights"]) - 1.0) < 1e-10


def test_best_model_highest_weight():
    preds = np.array([[1, 2], [3, 4]])
    log_ml = np.array([-100.0, -1.0])
    result = bayesian_model_averaging(preds, log_ml)
    assert result["top_model"] == 1
