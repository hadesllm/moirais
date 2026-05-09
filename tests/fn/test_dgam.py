"""Tests for moirais.fn.dgam — gamma PDF."""

import numpy as np
import pytest

from moirais.fn.dgam import dgamma


class TestDgamma:
    """Tests for dgamma()."""

    def test_exponential(self):
        """dgamma(1, shape=1, rate=1) = e^{-1} ~ 0.3679."""
        assert dgamma(1, shape=1, rate=1) == pytest.approx(0.3679, abs=1e-3)

    def test_at_zero(self):
        """dgamma(0, shape=1, rate=1) = 1.0 (exponential at origin)."""
        assert dgamma(0, shape=1, rate=1) == pytest.approx(1.0, abs=1e-6)

    def test_type(self):
        """Scalar input returns float."""
        result = dgamma(2.0, shape=2, rate=1)
        assert isinstance(result, (float, np.floating))

    def test_raises_nonpositive_shape(self):
        """Should reject shape <= 0."""
        with pytest.raises(ValueError):
            dgamma(1, shape=0)
