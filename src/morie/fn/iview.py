# morie.fn — function file (hadesllm/morie)
"""Image viewer and info. 'The garbage'll do!' -- Rey"""

from __future__ import annotations

import os
import struct
import subprocess
import sys

from ._containers import DescriptiveResult
from ._richresult import RichResult


def view_image(path: str) -> None:
    """Open an image in the system viewer.

    Parameters
    ----------
    path : str
        Path to image file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    if sys.platform == "darwin":
        subprocess.run(["open", path], check=True)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", path], check=True)
    else:
        os.startfile(path)


def image_info(path: str) -> dict:
    """Read basic image dimensions from PNG or JPEG header.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict
        Keys: width, height, format.
    """
    with open(path, "rb") as f:
        header = f.read(32)

    if header[:8] == b"\x89PNG\r\n\x1a\n":
        width = struct.unpack(">I", header[16:20])[0]
        height = struct.unpack(">I", header[20:24])[0]
        return RichResult(payload={"width": width, "height": height, "format": "PNG"})

    if header[:2] == b"\xff\xd8":
        with open(path, "rb") as f:
            f.read(2)
            while True:
                marker = f.read(2)
                if len(marker) < 2:
                    break
                if marker[0] != 0xFF:
                    break
                if marker[1] in (0xC0, 0xC1, 0xC2):
                    f.read(3)
                    h = struct.unpack(">H", f.read(2))[0]
                    w = struct.unpack(">H", f.read(2))[0]
                    return RichResult(payload={"width": w, "height": h, "format": "JPEG"})
                length = struct.unpack(">H", f.read(2))[0]
                f.read(length - 2)
        return RichResult(payload={"width": 0, "height": 0, "format": "JPEG"})

    return RichResult(payload={"width": 0, "height": 0, "format": "unknown"})


def image_info_result(path: str) -> DescriptiveResult:
    """Image info wrapped in DescriptiveResult."""
    info = image_info(path)
    return DescriptiveResult(name="Image info", value=info)


iview = image_info_result


def cheatsheet() -> str:
    return "view_image({}) -> Image viewer and info. 'The garbage'll do!' -- Rey"
