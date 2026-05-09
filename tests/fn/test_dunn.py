"""Test dunn."""
import numpy as np
import pytest
from moirais.fn.dunn import dunn_test


def test_dunn_basic():
    rng = np.random.default_rng(42)
    a = rng.standard_normal(20)
    b = rng.standard_normal(20) + 2
    c = rng.standard_normal(20) + 4
    r = dunn_test(a, b, c)
    assert r.name == "dunn"
    assert len(r.extra["pairs"]) == 3


def test_dunn_holm():
    rng = np.random.default_rng(7)
    a = rng.standard_normal(30)
    b = rng.standard_normal(30)
    r = dunn_test(a, b, method="holm")
    assert len(r.extra["adjusted_pvalues"]) == 1
