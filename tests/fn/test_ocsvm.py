"""Tests for morie.fn.ocsvm -- OC SVM classify."""
import numpy as np
from morie.fn.ocsvm import oc_svm_classify, ocsvm


def test_alias():
    assert ocsvm is oc_svm_classify


def test_smoke():
    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal(-1, 0.3, (20, 2)), rng.normal(1, 0.3, (20, 2))])
    labels = np.array([0]*20 + [1]*20, dtype=float)
    r = oc_svm_classify(X, labels)
    assert r.name == "oc_svm_classify"
    assert r.value > 0.7


def test_extra():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    labels = np.array([0, 0, 1, 1], dtype=float)
    r = oc_svm_classify(X, labels)
    assert "normal" in r.extra
    assert "accuracy" in r.extra
