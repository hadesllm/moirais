"""Tests for moirais.fn.ptmor -- Morisita index of dispersion"""

import numpy as np
import pytest

from moirais.fn.ptmor import pp_morisita


class TestPpMorisita:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pp_morisita(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = pp_morisita(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
