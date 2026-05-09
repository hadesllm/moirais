"""Tests for moirais.fn.lglbl -- label legislators."""
import numpy as np
from moirais.fn.lglbl import label_legislators, lglbl


def test_alias():
    assert lglbl is label_legislators


def test_smoke():
    X = np.array([[0.5], [-0.3], [0.1]])
    names = ["Smith", "Jones", "Doe"]
    r = label_legislators(X, names)
    assert r.name == "label_legislators"
    assert r.extra["n"] == 3
    assert len(r.extra["labeled"]) == 3
    assert r.extra["labeled"][0]["name"] == "Smith"
