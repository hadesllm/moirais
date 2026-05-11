"""Tests for morie.fn.iview -- Image viewer and info."""

import struct
import tempfile
import os
from unittest.mock import patch, MagicMock

from morie.fn.iview import view_image, image_info, iview


def _make_minimal_png(path: str) -> None:
    """Write a minimal valid 1x1 PNG file."""
    import zlib
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    raw = b"\x00\x00\x00\x00"
    compressed = zlib.compress(raw)
    idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
    idat = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    with open(path, "wb") as f:
        f.write(sig + ihdr + idat + iend)


class TestIview:
    def test_view_image_calls_open(self):
        with patch("morie.fn.iview.subprocess.run") as mock_run, \
             patch("morie.fn.iview.os.path.isfile", return_value=True):
            view_image("/tmp/fake.png")
            mock_run.assert_called_once()

    def test_image_info_png(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.png")
            _make_minimal_png(path)
            info = image_info(path)
            assert info["format"] == "PNG"
            assert info["width"] == 1
            assert info["height"] == 1

    def test_result_wrapper(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.png")
            _make_minimal_png(path)
            result = iview(path)
            assert result.value["format"] == "PNG"
