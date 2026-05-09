"""Tests for moirais.fn.tscls — 1-NN DTW classifier."""
import numpy as np
from moirais.fn.tscls import ts_classify


class TestTSClassify:
    def test_basic(self):
        X_train = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        y_train = np.array([0, 1, 0])
        X_test = np.array([[0, 1, 0.1]])
        res = ts_classify(X_train, y_train, X_test)
        assert res.extra["predictions"][0] == 0

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        X_tr = rng.standard_normal((5, 10))
        y_tr = np.array([0, 1, 0, 1, 0])
        X_te = rng.standard_normal((3, 10))
        res = ts_classify(X_tr, y_tr, X_te)
        assert len(res.extra["predictions"]) == 3
