"""Tests for moirais.fn.bmapr -- Bayesian model averaging."""

import numpy as np
from moirais.fn.bmapr import bayesian_model_averaging


def test_returns_dict():
    preds = [[1, 2, 3], [4, 5, 6]]
    log_ev = [-10, -12]
    result = bayesian_model_averaging(preds, log_ev)
    assert isinstance(result, dict)
    assert "averaged_predictions" in result
    assert "model_weights" in result


def test_weights_sum_to_one():
    preds = [[1, 2], [3, 4], [5, 6]]
    log_ev = [-10, -12, -11]
    result = bayesian_model_averaging(preds, log_ev)
    np.testing.assert_allclose(np.sum(result["model_weights"]), 1.0, atol=1e-10)


def test_dominant_model():
    preds = [[1, 2, 3], [10, 20, 30]]
    log_ev = [0, -1000]
    result = bayesian_model_averaging(preds, log_ev)
    np.testing.assert_allclose(result["averaged_predictions"], [1, 2, 3], atol=0.01)


def test_equal_evidence():
    preds = [[1, 2], [3, 4]]
    log_ev = [0, 0]
    result = bayesian_model_averaging(preds, log_ev)
    np.testing.assert_allclose(result["averaged_predictions"], [2, 3], atol=1e-10)
    np.testing.assert_allclose(result["model_weights"], [0.5, 0.5], atol=1e-10)


def test_prior_weights():
    preds = [[1, 2], [3, 4]]
    log_ev = [0, 0]
    result = bayesian_model_averaging(preds, log_ev, prior_weights=[3, 1])
    assert result["model_weights"][0] > result["model_weights"][1]


def test_mismatched_dims():
    try:
        bayesian_model_averaging([[1, 2], [3, 4], [5, 6]], [-10, -11])
        assert False
    except ValueError:
        pass
