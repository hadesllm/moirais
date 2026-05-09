# moirais.fn — function file (hadesllm/moirais)
"""Pi 5 system status for Perseus. 'When I let go of what I am, I become what I might be. — Lao Tzu'"""

from __future__ import annotations

from ._containers import DescriptiveResult


def pi_status() -> DescriptiveResult:
    """Return a full Raspberry Pi 5 system snapshot as a DescriptiveResult.

    Wraps :func:`moirais.sensors.pi_status` and :func:`moirais.sensors.pi_health_check`
    into the standard fn/ result format.
    """
    from moirais.sensors import pi_health_check
    from moirais.sensors import pi_status as _pi_status

    status = _pi_status()
    health = pi_health_check()

    return DescriptiveResult(
        name="pi_status",
        value=status,
        extra={
            "healthy": health["healthy"],
            "warnings": health["warnings"],
        },
    )


def cheatsheet() -> str:
    return "pi_status() -> Pi 5 system snapshot (temp, RAM, disk, Ollama, Hailo)"


pistat = pi_status
