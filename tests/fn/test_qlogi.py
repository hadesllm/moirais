"""Tests for moirais.fn.qlogi -- Logistic quantile."""

import numpy as np
import pytest
from moirais.fn.qlogi import qlogi


class TestQlogi:
    def test_median(self):
        """qlogi(0.5) = 0 for standard logistic."""
        assert qlogi(0.5) == pytest.approx(0.0, abs=1e-10)

    def test_known_quantile(self):
        """qlogi(0.75) = ln(3) ~ 1.0986."""
        assert qlogi(0.75) == pytest.approx(np.log(3), abs=1e-4)

    def test_nonstandard_median(self):
        """Shifted logistic median equals loc."""
        assert qlogi(0.5, loc=5.0, scale=2.0) == pytest.approx(5.0, abs=1e-10)

    def test_raises_bad_scale(self):
        with pytest.raises(ValueError):
            qlogi(0.5, scale=0)
