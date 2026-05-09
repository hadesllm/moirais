"""Raspberry Pi 5 hardware sensor awareness for Perseus.

Provides system status, health checks, and thermal monitoring.
Degrades gracefully on non-Pi systems (returns None / 'N/A').
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone

try:
    import psutil
except ImportError:
    psutil = None

try:
    import requests
except ImportError:
    requests = None


def _run_cmd(cmd: list[str], timeout: float = 5.0) -> str | None:
    """Run a shell command, return stdout or None on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _vcgencmd(*args: str) -> str | None:
    return _run_cmd(["vcgencmd", *args])


def _cpu_temp() -> float | None:
    """Read CPU temperature via vcgencmd."""
    raw = _vcgencmd("measure_temp")
    if raw:
        try:
            return float(raw.replace("temp=", "").replace("'C", ""))
        except (ValueError, AttributeError):
            pass
    if psutil is not None:
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for entries in temps.values():
                    if entries:
                        return float(entries[0].current)
        except (AttributeError, RuntimeError):
            pass
    return None


def _gpu_mem() -> str | None:
    raw = _vcgencmd("get_mem", "gpu")
    if raw:
        return raw.replace("gpu=", "")
    return None


def _hailo_status() -> dict:
    """Check Hailo NPU device presence."""
    import glob as _glob

    devices = _glob.glob("/dev/hailo*")
    return {"detected": len(devices) > 0, "devices": devices}


def _ollama_models() -> list[str] | None:
    """Query local Ollama for installed models."""
    if requests is None:
        return None
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
    except (requests.ConnectionError, requests.Timeout, ValueError, KeyError):
        pass
    return None


def _network_interfaces() -> dict[str, list[str]]:
    """Return interface -> list of IP addresses."""
    if psutil is None:
        return {}
    try:
        addrs = psutil.net_if_addrs()
        result = {}
        for iface, entries in addrs.items():
            ips = []
            for e in entries:
                if e.family.name in ("AF_INET", "AF_INET6"):
                    ips.append(e.address)
            if ips:
                result[iface] = ips
        return result
    except (AttributeError, RuntimeError):
        return {}


def _uptime() -> str | None:
    if psutil is None:
        return None
    try:
        boot = psutil.boot_time()
        delta = time.time() - boot
        days = int(delta // 86400)
        hours = int((delta % 86400) // 3600)
        mins = int((delta % 3600) // 60)
        return f"{days}d {hours}h {mins}m"
    except (AttributeError, RuntimeError):
        return None


def pi_status() -> dict:
    """Full system snapshot of the Raspberry Pi 5 environment.

    Returns a dict with keys: cpu_temp, cpu_percent, ram, swap, disk,
    uptime, load_avg, gpu_mem, network, ollama_models, hailo.
    Works on non-Pi systems with degraded output (None for Pi-specific items).
    """
    status: dict = {}

    status["timestamp"] = datetime.now(timezone.utc).isoformat()
    status["cpu_temp_c"] = _cpu_temp()

    if psutil is not None:
        try:
            status["cpu_percent"] = psutil.cpu_percent(interval=0.5)
        except (AttributeError, RuntimeError):
            status["cpu_percent"] = None

        try:
            mem = psutil.virtual_memory()
            status["ram"] = {
                "total_gb": round(mem.total / (1024**3), 2),
                "used_gb": round(mem.used / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "percent": mem.percent,
            }
        except (AttributeError, RuntimeError):
            status["ram"] = None

        try:
            sw = psutil.swap_memory()
            status["swap"] = {
                "total_gb": round(sw.total / (1024**3), 2),
                "used_gb": round(sw.used / (1024**3), 2),
                "percent": sw.percent,
            }
        except (AttributeError, RuntimeError):
            status["swap"] = None

        disks = {}
        for mount in ["/", "/mnt/nvme"]:
            try:
                du = psutil.disk_usage(mount)
                disks[mount] = {
                    "total_gb": round(du.total / (1024**3), 2),
                    "used_gb": round(du.used / (1024**3), 2),
                    "free_gb": round(du.free / (1024**3), 2),
                    "percent": du.percent,
                }
            except (FileNotFoundError, OSError):
                disks[mount] = None
        status["disk"] = disks
    else:
        status["cpu_percent"] = None
        status["ram"] = None
        status["swap"] = None
        status["disk"] = {"/": None, "/mnt/nvme": None}

    status["uptime"] = _uptime()

    try:
        import os

        status["load_avg"] = list(os.getloadavg())
    except (AttributeError, OSError):
        status["load_avg"] = None

    status["gpu_mem"] = _gpu_mem()
    status["network"] = _network_interfaces()
    status["ollama_models"] = _ollama_models()
    status["hailo"] = _hailo_status()

    return status


def pi_health_check() -> dict:
    """Run health checks and return a dict with 'healthy' bool and 'warnings' list.

    Checks:
    - CPU temperature > 80C
    - RAM available < 1 GB
    - Swap usage > 50%
    - NVMe disk > 90% full
    - Ollama not running
    """
    warnings: list[str] = []

    temp = _cpu_temp()
    if temp is not None and temp > 80.0:
        warnings.append(f"CPU temperature critical: {temp:.1f}C (> 80C)")

    if psutil is not None:
        try:
            mem = psutil.virtual_memory()
            avail_gb = mem.available / (1024**3)
            if avail_gb < 1.0:
                warnings.append(f"RAM low: {avail_gb:.2f} GB available (< 1 GB)")
        except (AttributeError, RuntimeError):
            pass

        try:
            sw = psutil.swap_memory()
            if sw.total > 0 and sw.percent > 50.0:
                warnings.append(f"Swap high: {sw.percent:.1f}% used (> 50%)")
        except (AttributeError, RuntimeError):
            pass

        try:
            du = psutil.disk_usage("/mnt/nvme")
            if du.percent > 90.0:
                warnings.append(f"NVMe nearly full: {du.percent:.1f}% used (> 90%)")
        except (FileNotFoundError, OSError):
            pass

    models = _ollama_models()
    if models is None:
        warnings.append("Ollama not running or not reachable on localhost:11434")

    return {"healthy": len(warnings) == 0, "warnings": warnings}


def pi_thermal_history(seconds: int = 60) -> list[tuple[str, float | None]]:
    """Sample CPU temperature every 2 seconds for *seconds* duration.

    Returns a list of (ISO timestamp, temperature_C) tuples.
    """
    samples: list[tuple[str, float | None]] = []
    end_time = time.time() + seconds
    while time.time() < end_time:
        ts = datetime.now(timezone.utc).isoformat()
        temp = _cpu_temp()
        samples.append((ts, temp))
        remaining = end_time - time.time()
        if remaining > 2.0:
            time.sleep(2.0)
        elif remaining > 0:
            time.sleep(remaining)
            break
        else:
            break
    return samples


def format_status(status: dict) -> str:
    """Pretty-print a pi_status() dict as a human-readable report."""
    lines: list[str] = []
    lines.append("=== MOIRAIS Pi 5 System Status ===")
    lines.append(f"Timestamp : {status.get('timestamp', 'N/A')}")
    lines.append("")

    temp = status.get("cpu_temp_c")
    lines.append(f"CPU Temp  : {f'{temp:.1f} C' if temp is not None else 'N/A'}")
    cpu_pct = status.get("cpu_percent")
    lines.append(f"CPU Usage : {f'{cpu_pct:.1f}%' if cpu_pct is not None else 'N/A'}")

    load = status.get("load_avg")
    if load:
        lines.append(f"Load Avg  : {load[0]:.2f}  {load[1]:.2f}  {load[2]:.2f}")
    else:
        lines.append("Load Avg  : N/A")

    lines.append(f"Uptime    : {status.get('uptime', 'N/A')}")
    lines.append(f"GPU Mem   : {status.get('gpu_mem', 'N/A')}")
    lines.append("")

    ram = status.get("ram")
    if ram:
        lines.append(
            f"RAM       : {ram['used_gb']:.1f} / {ram['total_gb']:.1f} GB "
            f"({ram['percent']:.0f}%) -- {ram['available_gb']:.1f} GB free"
        )
    else:
        lines.append("RAM       : N/A")

    swap = status.get("swap")
    if swap:
        lines.append(f"Swap      : {swap['used_gb']:.1f} / {swap['total_gb']:.1f} GB ({swap['percent']:.0f}%)")
    else:
        lines.append("Swap      : N/A")
    lines.append("")

    disk = status.get("disk", {})
    for mount, info in disk.items():
        if info:
            lines.append(
                f"Disk {mount:12s}: {info['used_gb']:.1f} / {info['total_gb']:.1f} GB "
                f"({info['percent']:.0f}%) -- {info['free_gb']:.1f} GB free"
            )
        else:
            lines.append(f"Disk {mount:12s}: N/A")
    lines.append("")

    hailo = status.get("hailo", {})
    if hailo.get("detected"):
        lines.append(f"Hailo NPU : detected ({', '.join(hailo['devices'])})")
    else:
        lines.append("Hailo NPU : not detected")

    models = status.get("ollama_models")
    if models is not None:
        lines.append(
            f"Ollama    : {len(models)} model(s) -- {', '.join(models[:5])}" + (" ..." if len(models) > 5 else "")
        )
    else:
        lines.append("Ollama    : not running")
    lines.append("")

    net = status.get("network", {})
    if net:
        for iface, ips in net.items():
            lines.append(f"Net {iface:8s}: {', '.join(ips)}")
    else:
        lines.append("Network   : N/A")

    lines.append("=" * 35)
    return "\n".join(lines)
