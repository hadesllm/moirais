"""Tests for moirais.fn.nmala -- Alpha-NOMINATE acceptance rate"""

import numpy as np
import pytest

from moirais.fn.nmala import alpha_nom_accept


class TestAlphaNomAccept:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = alpha_nom_accept(data)
        assert result.value is not None

    def test_output_type(self):
        result = alpha_nom_accept(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
