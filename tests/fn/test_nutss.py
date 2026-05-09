"""Tests for moirais.fn.nutss -- No-U-Turn Sampler."""

import numpy as np
from moirais.fn.nutss import nuts_sampler


def _log_normal(x):
    return -0.5 * float(x @ x)


def _grad_normal(x):
    return -x


def test_returns_dict():
    result = nuts_sampler(_log_normal, _grad_normal, [0.0, 0.0], n_iter=50)
    assert isinstance(result, dict)
    assert "samples" in result
    assert "mean_tree_depth" in result


def test_correct_shape():
    result = nuts_sampler(_log_normal, _grad_normal, [0.0], n_iter=100)
    assert result["samples"].shape == (100, 1)


def test_samples_near_target_mean():
    result = nuts_sampler(
        _log_normal, _grad_normal, [0.0, 0.0], n_iter=500, epsilon=0.1
    )
    means = np.mean(result["samples"][100:], axis=0)
    assert np.all(np.abs(means) < 1.0)


def test_tree_depth_positive():
    result = nuts_sampler(_log_normal, _grad_normal, [0.0], n_iter=50)
    assert result["mean_tree_depth"] > 0


def test_invalid_epsilon():
    try:
        nuts_sampler(_log_normal, _grad_normal, [0.0], epsilon=-1)
        assert False
    except ValueError:
        pass
