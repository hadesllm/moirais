"""Tests for morie.fn.suebac -- eBAC distribution."""

import pytest
from morie.fn.suebac import ebac_dist


class TestEbacDist:
    def test_basic(self):
        res = ebac_dist([0.02, 0.05, 0.08, 0.10, 0.12])
        assert res.name == "ebac_distribution"
        assert res.extra["pct_over_limit"] == pytest.approx(40.0)

    def test_all_under(self):
        res = ebac_dist([0.01, 0.02, 0.03])
        assert res.extra["pct_over_limit"] == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ebac_dist([])
