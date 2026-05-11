"""Test parzen_pdf."""
import numpy as np
from morie.fn.przpd import parzen_pdf, przpd
from morie.fn._containers import DescriptiveResult


class TestParzenPDF:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(200)
        result = parzen_pdf(x)
        assert isinstance(result, DescriptiveResult)

    def test_density_nonneg(self):
        x = np.random.default_rng(42).standard_normal(200)
        result = parzen_pdf(x)
        assert np.all(result.extra["density"] >= 0)

    def test_grid_length(self):
        x = np.random.default_rng(42).standard_normal(200)
        result = parzen_pdf(x, n_points=50)
        assert len(result.extra["grid"]) == 50

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(200)
        result = parzen_pdf(x)
        assert result.name == "parzen_pdf"

    def test_alias(self):
        assert przpd is parzen_pdf
