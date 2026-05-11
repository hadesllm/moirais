"""Tests for morie.fn.yllst -- YLL with discounting."""

import pytest
from morie.fn.yllst import years_of_life_lost_std


class TestYLLStd:
    def test_basic(self):
        res = years_of_life_lost_std(
            deaths=[10], ages_at_death=[50], life_expectancy=80, discount_rate=0.0
        )
        assert res.estimate == pytest.approx(300.0)

    def test_discounted_less(self):
        no_disc = years_of_life_lost_std(
            deaths=[10], ages_at_death=[50], life_expectancy=80, discount_rate=0.0
        )
        with_disc = years_of_life_lost_std(
            deaths=[10], ages_at_death=[50], life_expectancy=80, discount_rate=0.03
        )
        assert with_disc.estimate < no_disc.estimate

    def test_mismatch(self):
        with pytest.raises(ValueError):
            years_of_life_lost_std(deaths=[10], ages_at_death=[50, 60])
