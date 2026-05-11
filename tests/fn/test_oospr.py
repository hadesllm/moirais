"""Test oos_predict."""
import numpy as np
from morie.fn.oospr import oos_predict, oospr
from morie.fn._containers import DescriptiveResult


class TestOosPredict:
    def test_basic(self):
        def model_fn(X_train, y_train, X_test):
            return np.mean(y_train) * np.ones(X_test.shape[0])

        X_tr = np.array([[1], [2], [3]])
        y_tr = np.array([1.0, 2.0, 3.0])
        X_te = np.array([[4], [5]])
        result = oos_predict(model_fn, X_tr, y_tr, X_te)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "oos_predict"

    def test_prediction_count(self):
        def model_fn(X_train, y_train, X_test):
            return np.zeros(X_test.shape[0])

        result = oos_predict(model_fn, np.zeros((5, 2)), np.zeros(5), np.zeros((3, 2)))
        assert result.extra["n_test"] == 3

    def test_alias(self):
        assert oospr is oos_predict
