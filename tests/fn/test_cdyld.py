"""Tests for moirais.fn.cdyld -- YLD."""

import pytest
from moirais.fn.cdyld import years_lived_disability


class TestYLD:
    def test_basic(self):
        res = years_lived_disability(incidence=100, duration=5, disability_weight=0.2)
        assert res.estimate == pytest.approx(100.0)

    def test_invalid_weight(self):
        with pytest.raises(ValueError):
            years_lived_disability(100, 5, 1.5)
