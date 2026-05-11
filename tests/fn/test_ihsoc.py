"""Tests for morie.fn.ihsoc -- social determinants."""

import pytest
from morie.fn.ihsoc import social_determinants


class TestSocialDeterminants:
    def test_equal_weights(self):
        res = social_determinants([0.5, 0.7, 0.3, 0.9])
        assert res.estimate == pytest.approx(0.6)

    def test_weighted(self):
        res = social_determinants([0.5, 1.0], weights=[1, 3])
        assert res.estimate == pytest.approx(0.875)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            social_determinants([])
