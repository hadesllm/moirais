"""Tests for morie.fn.stpwz -- Stepped-wedge design."""

import pytest
from morie.fn.stpwz import stepped_wedge_design


class TestSteppedWedge:
    def test_basic(self):
        res = stepped_wedge_design(clusters=20, steps=4, cluster_size=50, effect_size=0.1)
        assert res.measure == "stepped_wedge"
        assert 0 <= res.estimate <= 1

    def test_more_clusters_more_power(self):
        few = stepped_wedge_design(clusters=10, steps=4, cluster_size=50, effect_size=0.1)
        many = stepped_wedge_design(clusters=40, steps=4, cluster_size=50, effect_size=0.1)
        assert many.estimate >= few.estimate

    def test_invalid(self):
        with pytest.raises(ValueError):
            stepped_wedge_design(clusters=1, steps=2, cluster_size=10, effect_size=0.1)
