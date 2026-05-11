"""Tests for morie.fn.savfg -- Save figure."""

import pytest
from unittest.mock import MagicMock

from morie.fn.savfg import save_figure, savfg, _VALID_FORMATS
from morie.fn._containers import DescriptiveResult


class TestSavfg:
    def test_alias(self):
        assert savfg is save_figure

    def test_invalid_format(self):
        fig = MagicMock()
        with pytest.raises(ValueError, match="Format"):
            save_figure(fig, "/tmp/test.bmp", fmt="bmp")

    def test_valid_format_logic(self):
        assert "png" in _VALID_FORMATS
        assert "pdf" in _VALID_FORMATS
        assert "svg" in _VALID_FORMATS

    def test_calls_savefig(self, tmp_path):
        fig = MagicMock()
        path = str(tmp_path / "output.png")
        result = save_figure(fig, path, dpi=100, fmt="png")
        assert isinstance(result, DescriptiveResult)
        fig.savefig.assert_called_once()
        assert result.extra["dpi"] == 100
