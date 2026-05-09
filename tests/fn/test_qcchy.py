"""Tests for moirais.fn.qcchy -- Cauchy quantile."""

import numpy as np
import pytest
from moirais.fn.qcchy import qcchy


class TestQcchy:
    def test_median(self):
        """qcchy(0.5) = loc = 0 for standard Cauchy."""
        assert qcchy(0.5) == pytest.approx(0.0, abs=1e-10)

    def test_quartiles(self):
        """qcchy(0.25) = -1 and qcchy(0.75) = 1 for standard Cauchy."""
        assert qcchy(0.25) == pytest.approx(-1.0, abs=1e-10)
        assert qcchy(0.75) == pytest.approx(1.0, abs=1e-10)

    def test_nonstandard(self):
        """Shifted Cauchy median equals loc."""
        assert qcchy(0.5, loc=5.0, scale=2.0) == pytest.approx(5.0, abs=1e-10)

    def test_raises_bad_scale(self):
        with pytest.raises(ValueError):
            qcchy(0.5, scale=0)
