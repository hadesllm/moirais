"""Tests for moirais.fn.mdslb -- label MDS points."""

import numpy as np
from moirais.fn.mdslb import label_mds_points, mdslb


def test_mdslb_smoke():
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    labels = ["A", "B", "C"]
    r = mdslb(X, labels)
    assert r.name == "label_mds_points"
    assert "A" in r.value
    assert r.value["A"] == [1.0, 2.0]


def test_mdslb_alias():
    assert mdslb is label_mds_points
