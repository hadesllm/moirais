# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bootstrap CI. 'Bee-boop!' -- BB-8"""

from moirais.fn.boot import bootstrap_ci as _boot

bb8 = _boot
bootstrap_ci = _boot


def cheatsheet() -> str:
    return "bb8() -> Bootstrap CI. 'Bee-boop!' -- BB-8"
