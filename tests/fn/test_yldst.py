"""Tests for morie.fn.yldst -- Years lived with disability."""

import pytest
from morie.fn.yldst import years_lived_with_disability


class TestYLD:
    def test_basic(self):
        res = years_lived_with_disability(
            cases=[100], durations=[10.0], weights=[0.5], discount_rate=0.0
        )
        assert res.estimate == pytest.approx(500.0)

    def test_discounted(self):
        no_disc = years_lived_with_disability(
            cases=[100], durations=[10.0], weights=[0.5], discount_rate=0.0
        )
        disc = years_lived_with_disability(
            cases=[100], durations=[10.0], weights=[0.5], discount_rate=0.03
        )
        assert disc.estimate < no_disc.estimate

    def test_invalid_weight(self):
        with pytest.raises(ValueError):
            years_lived_with_disability(cases=[10], durations=[5], weights=[1.5])
