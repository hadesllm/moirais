"""Tests for morie.fn.nb_ -- Gaussian Naive Bayes."""

import numpy as np
from morie.fn.nb_ import naive_bayes, naive_bayes_fit, naive_bayes_predict, nb_
from morie.fn._containers import DescriptiveResult


class TestNb:
    def test_alias(self):
        assert nb_ is naive_bayes

    def test_separable_data(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.5, (50, 2)), rng.normal(3, 0.5, (50, 2))])
        y = np.array([0] * 50 + [1] * 50)
        result = naive_bayes(X, y)
        assert isinstance(result, DescriptiveResult)
        assert result.value > 0.8

    def test_fit_predict(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(-2, 1, (30, 2)), rng.normal(2, 1, (30, 2))])
        y = np.array([0] * 30 + [1] * 30)
        model = naive_bayes_fit(X, y)
        preds = naive_bayes_predict(model, X)
        assert len(preds) == 60
