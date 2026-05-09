"""Tests for moirais.fn.yll -- Years of Life Lost."""

import numpy as np
import pytest
from moirais.fn.yll import years_of_life_lost


class TestYLL:
    def test_known_yll(self):
        """Deaths at 70 with LE=80 -> 10 YLL each."""
        result = years_of_life_lost([70, 70, 70], life_expectancy=80)
        assert result["total_yll"] == pytest.approx(30.0)
        assert result["mean_yll"] == pytest.approx(10.0)
        assert result["n"] == 3

    def test_death_above_le_zero_yll(self):
        """Death at 90 with LE=80 -> 0 YLL."""
        result = years_of_life_lost([90], life_expectancy=80)
        assert result["total_yll"] == pytest.approx(0.0)

    def test_mixed_ages(self):
        result = years_of_life_lost([50, 60, 85], life_expectancy=80)
        assert result["total_yll"] == pytest.approx(30 + 20 + 0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            years_of_life_lost([])
