"""Pure-Python emissions tracker — drop-in replacement for codecarbon.

Replicates the CodeCarbon EmissionsTracker methodology without pydantic
or any Rust dependencies, enabling Python 3.15+ support.

Methodology (matches codecarbon 3.2.x):
  - CPU/GPU power: ``powermetrics`` on Apple Silicon, ``/proc/stat`` TDP on Linux
  - RAM power: heuristic based on DIMM count (1.5 W/DIMM ARM, 5 W/DIMM x86)
  - Carbon intensity: per-country energy mix → weighted gCO2/kWh
  - Emissions: ``energy_kWh × carbon_intensity_kgCO2_per_kWh × PUE``
  - Water: ``energy_kWh × WUE``

References:
  - CodeCarbon: https://github.com/mlco2/codecarbon (MIT license)
  - Global energy mix data: IEA / Our World in Data
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_JOULES_TO_KWH = 2.77778e-7
_G_TO_KG = 0.001
_LBS_MWH_TO_KG_KWH = 0.00045359237

# Data directory containing energy mix and carbon intensity files
_DATA_DIR = Path(__file__).parent / "data"

# Carbon intensity per source (gCO2/kWh) — matches codecarbon / IEA
_SOURCE_INTENSITY: dict[str, float] = {}
_WORLD_AVERAGE_G_KWH = 475.0

# Loaded lazily from JSON files
_GLOBAL_ENERGY_MIX: dict[str, Any] = {}
_ENERGY_MIX_LOADED = False


def _load_energy_data() -> None:
    """Load global energy mix and carbon intensity data from JSON files."""
    global _SOURCE_INTENSITY, _GLOBAL_ENERGY_MIX, _ENERGY_MIX_LOADED
    if _ENERGY_MIX_LOADED:
        return
    _ENERGY_MIX_LOADED = True

    # Carbon intensity per source
    src_path = _DATA_DIR / "carbon_intensity_per_source.json"
    if src_path.exists():
        import json

        with open(src_path) as f:
            _SOURCE_INTENSITY.update(json.load(f))
    else:
        _SOURCE_INTENSITY.update(
            {
                "coal": 995,
                "petroleum": 816,
                "natural_gas": 743,
                "fossil": 635,
                "geothermal": 38,
                "hydroelectricity": 26,
                "nuclear": 29,
                "solar": 48,
                "wind": 26,
            }
        )

    # Global energy mix (213 countries)
    mix_path = _DATA_DIR / "global_energy_mix.json"
    if mix_path.exists():
        import json

        with open(mix_path) as f:
            _GLOBAL_ENERGY_MIX.update(json.load(f))


# RAM power estimation
_RAM_POWER_PER_DIMM_ARM = 1.5  # Watts
_RAM_POWER_PER_DIMM_X86 = 5.0
_RAM_MIN_ARM = 3.0
_RAM_MIN_X86 = 10.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EmissionsData:
    """All tracked emissions data for a single run."""

    timestamp: str = ""
    project_name: str = "morie"
    run_id: str = ""
    experiment_id: str = ""
    duration: float = 0.0
    emissions: float = 0.0  # kg CO2
    emissions_rate: float = 0.0  # kg CO2/s
    cpu_power: float = 0.0  # Watts (average)
    gpu_power: float = 0.0  # Watts (average)
    ram_power: float = 0.0  # Watts
    cpu_energy: float = 0.0  # kWh
    gpu_energy: float = 0.0  # kWh
    ram_energy: float = 0.0  # kWh
    energy_consumed: float = 0.0  # kWh total
    water_consumed: float = 0.0  # litres
    country_name: str = ""
    country_iso_code: str = ""
    region: str = ""
    os_info: str = ""
    python_version: str = ""
    tracker_version: str = "morie-1.0.0"
    cpu_count: int = 0
    cpu_model: str = ""
    gpu_count: int = 0
    gpu_model: str = ""
    longitude: float = 0.0
    latitude: float = 0.0
    ram_total_size: float = 0.0  # GB
    tracking_mode: str = "machine"
    cpu_utilization_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    ram_utilization_percent: float = 0.0
    ram_used_gb: float = 0.0
    on_cloud: str = "N"
    pue: float = 1.0
    wue: float = 0.0
    cloud_provider: str = ""
    cloud_region: str = ""

    @property
    def csv_header(self) -> str:
        return (
            "timestamp,project_name,run_id,experiment_id,duration,"
            "emissions,emissions_rate,cpu_power,gpu_power,ram_power,"
            "cpu_energy,gpu_energy,ram_energy,energy_consumed,water_consumed,"
            "country_name,country_iso_code,region,os,python_version,"
            "codecarbon_version,cpu_count,cpu_model,gpu_count,gpu_model,"
            "longitude,latitude,ram_total_size,tracking_mode,"
            "cpu_utilization_percent,gpu_utilization_percent,"
            "ram_utilization_percent,ram_used_gb,on_cloud,pue,wue,"
            "cloud_provider,cloud_region"
        )

    def csv_row(self) -> str:
        return ",".join(
            str(v)
            for v in [
                self.timestamp,
                self.project_name,
                self.run_id,
                self.experiment_id,
                self.duration,
                self.emissions,
                self.emissions_rate,
                self.cpu_power,
                self.gpu_power,
                self.ram_power,
                self.cpu_energy,
                self.gpu_energy,
                self.ram_energy,
                self.energy_consumed,
                self.water_consumed,
                self.country_name,
                self.country_iso_code,
                self.region,
                self.os_info,
                self.python_version,
                self.tracker_version,
                self.cpu_count,
                self.cpu_model,
                self.gpu_count,
                self.gpu_model,
                self.longitude,
                self.latitude,
                self.ram_total_size,
                self.tracking_mode,
                self.cpu_utilization_percent,
                self.gpu_utilization_percent,
                self.ram_utilization_percent,
                self.ram_used_gb,
                self.on_cloud,
                self.pue,
                self.wue,
                self.cloud_provider,
                self.cloud_region,
            ]
        )


# ---------------------------------------------------------------------------
# Hardware measurement helpers
# ---------------------------------------------------------------------------


def _is_apple_silicon() -> bool:
    return platform.machine() == "arm64" and sys.platform == "darwin"


def _estimate_ram_power() -> float:
    """Estimate RAM power draw (Watts) using codecarbon heuristic."""
    import psutil

    total_gb = psutil.virtual_memory().total / (1024**3)
    is_arm = platform.machine() in ("arm64", "aarch64")
    base = _RAM_POWER_PER_DIMM_ARM if is_arm else _RAM_POWER_PER_DIMM_X86
    minimum = _RAM_MIN_ARM if is_arm else _RAM_MIN_X86

    # Estimate DIMM count
    if total_gb <= 8 or total_gb <= 16:
        dimms = 2
    elif total_gb <= 32 or total_gb <= 64:
        dimms = 4
    elif total_gb <= 128:
        dimms = 8
    else:
        dimms = min(int(total_gb / 16), 32)

    # Apply efficiency scaling (codecarbon methodology)
    if dimms <= 4:
        power = base * dimms
    elif dimms <= 8:
        power = (base * 4) + (base * 0.9 * (dimms - 4))
    elif dimms <= 16:
        power = (base * 4) + (base * 0.9 * 4) + (base * 0.8 * (dimms - 8))
    else:
        power = (base * 4) + (base * 0.9 * 4) + (base * 0.8 * 8) + (base * 0.7 * (dimms - 16))

    return max(power, minimum)


def _read_powermetrics(duration_ms: int = 500) -> tuple[float, float]:
    """Read CPU and GPU power via macOS powermetrics (mW → W).

    Returns (cpu_watts, gpu_watts).  Falls back to (0, 0) if powermetrics
    is unavailable or requires sudo.
    """
    try:
        result = subprocess.run(
            ["powermetrics", "-n", "1", "--samplers", "cpu_power", "-i", str(duration_ms)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        text = result.stdout + result.stderr
        cpu_match = re.search(r"CPU Power:\s*(\d+)\s*mW", text)
        gpu_match = re.search(r"GPU Power:\s*(\d+)\s*mW", text)
        cpu_w = int(cpu_match.group(1)) / 1000.0 if cpu_match else 0.0
        gpu_w = int(gpu_match.group(1)) / 1000.0 if gpu_match else 0.0
        return cpu_w, gpu_w
    except Exception:
        return 0.0, 0.0


from functools import lru_cache


@lru_cache(maxsize=1)
def _cpu_tdp_fallback() -> float:
    """Estimate CPU TDP from model name (Watts). Cached — CPU doesn't change."""
    # Try sysctl on macOS first (most reliable for Apple Silicon)
    brand = ""
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            brand = result.stdout.strip()
        except Exception:
            pass

    if not brand:
        try:
            import cpuinfo

            info = cpuinfo.get_cpu_info()
            brand = info.get("brand_raw", "")
        except Exception:
            brand = platform.processor()

    brand_lower = brand.lower()
    # Apple Silicon (sysctl returns "Apple M1/M2/M3/M4")
    if "m4" in brand_lower or "m3" in brand_lower:
        return 12.0
    elif "m2" in brand_lower:
        return 15.0
    elif "m1" in brand_lower:
        return 10.0
    # Also detect via platform.processor() == "arm" on macOS
    elif _is_apple_silicon():
        return 12.0  # conservative for unknown Apple Silicon
    elif "i9" in brand_lower:
        return 125.0
    elif "i7" in brand_lower or "i5" in brand_lower:
        return 65.0
    elif "ryzen 9" in brand_lower:
        return 105.0
    elif "ryzen 7" in brand_lower:
        return 65.0
    elif "xeon" in brand_lower:
        return 150.0
    return 85.0  # conservative default


def _get_carbon_intensity(
    country_iso: str = "",
    region: str = "",
) -> float:
    """Return carbon intensity in kgCO2/kWh for a given location.

    Uses the same data sources as codecarbon:
    - 213 countries from IEA global energy mix
    - US state-level data (EPA eGRID)
    - Canadian provincial data
    """
    _load_energy_data()

    # US state-level data
    if country_iso == "USA" and region:
        usa_path = _DATA_DIR / "2016" / "usa_emissions.json"
        if usa_path.exists():
            import json

            with open(usa_path) as f:
                usa_data = json.load(f)
            for state, info in usa_data.items():
                if region.lower() in state.lower() or state.lower() in region.lower():
                    lbs_mwh = info if isinstance(info, (int, float)) else info.get("emissions", 0)
                    return float(lbs_mwh) * _LBS_MWH_TO_KG_KWH

    # Canadian provincial data
    if country_iso == "CAN" and region:
        can_path = _DATA_DIR / "2023" / "canada_energy_mix.json"
        if can_path.exists():
            import json

            with open(can_path) as f:
                can_data = json.load(f)
            for prov, info in can_data.items():
                if region.lower() in prov.lower() or prov.lower() in region.lower():
                    if isinstance(info, (int, float)):
                        return float(info) * _G_TO_KG
                    elif isinstance(info, dict) and "carbon_intensity" in info:
                        return float(info["carbon_intensity"]) * _G_TO_KG

    # Global energy mix (213 countries)
    if country_iso in _GLOBAL_ENERGY_MIX:
        country_data = _GLOBAL_ENERGY_MIX[country_iso]
        if isinstance(country_data, dict):
            # Direct carbon intensity
            if "carbon_intensity" in country_data:
                return float(country_data["carbon_intensity"]) * _G_TO_KG
            # Compute from energy mix
            total_twh = 0.0
            weighted_intensity = 0.0
            for source, twh in country_data.items():
                if source in _SOURCE_INTENSITY and isinstance(twh, (int, float)):
                    total_twh += twh
                    weighted_intensity += twh * _SOURCE_INTENSITY[source]
            if total_twh > 0:
                return (weighted_intensity / total_twh) * _G_TO_KG

    return _WORLD_AVERAGE_G_KWH * _G_TO_KG


def _detect_location() -> tuple[str, str, str, float, float]:
    """Detect country, region, name, lat, lon via IP geolocation.

    Returns (iso_code, region, country_name, latitude, longitude).
    """
    try:
        import httpx

        resp = httpx.get("https://ipapi.co/json/", timeout=5)
        data = resp.json()
        return (
            data.get("country_code", ""),
            data.get("region", ""),
            data.get("country_name", ""),
            float(data.get("latitude", 0)),
            float(data.get("longitude", 0)),
        )
    except Exception:
        return ("", "", "", 0.0, 0.0)


def _get_cpu_model() -> str:
    try:
        if sys.platform == "darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return result.stdout.strip() or platform.processor()
        return platform.processor()
    except Exception:
        return platform.processor()


# ---------------------------------------------------------------------------
# EmissionsTracker
# ---------------------------------------------------------------------------


class EmissionsTracker:
    """Pure-Python emissions tracker compatible with codecarbon's API.

    Usage::

        tracker = EmissionsTracker(project_name="my-analysis")
        tracker.start()
        # ... run computation ...
        emissions_kg = tracker.stop()

    Context manager::

        with EmissionsTracker() as tracker:
            ...  # tracked
    """

    def __init__(
        self,
        project_name: str = "morie",
        output_dir: str = ".",
        output_file: str = "emissions.csv",
        measure_power_secs: float = 15.0,
        tracking_mode: str = "machine",
        pue: float = 1.0,
        wue: float = 0.0,
        log_level: str = "warning",
        save_to_file: bool = True,
        save_to_logger: bool = False,
        country_iso_code: str = "",
        region: str = "",
        **kwargs: Any,
    ) -> None:
        self._project_name = project_name
        self._output_dir = Path(output_dir)
        self._output_file = output_file
        self._measure_interval = measure_power_secs
        self._tracking_mode = tracking_mode
        self._pue = pue
        self._wue = wue
        self._save_to_file = save_to_file
        self._country_iso = country_iso_code
        self._region = region

        self._run_id = str(uuid.uuid4())
        self._experiment_id = kwargs.get("experiment_id", str(uuid.uuid4()))
        self._start_time: float = 0.0
        self._total_cpu_energy: float = 0.0
        self._total_gpu_energy: float = 0.0
        self._total_ram_energy: float = 0.0
        self._cpu_power_samples: list[float] = []
        self._gpu_power_samples: list[float] = []
        self._cpu_util_samples: list[float] = []
        self._ram_util_samples: list[float] = []
        self._ram_used_samples: list[float] = []
        self._ram_power: float = 0.0
        self._last_measure_time: float = 0.0
        self._stop_event = Event()
        self._bg_thread: Thread | None = None
        self._is_apple = _is_apple_silicon()

        # Location
        self._country_name = ""
        self._longitude = 0.0
        self._latitude = 0.0
        # Cached at start() to avoid recomputing per-stop
        self._carbon_intensity: float = 0.0
        self._tdp: float = 0.0

    def start(self) -> EmissionsTracker:

        self._start_time = time.time()
        self._last_measure_time = self._start_time
        self._ram_power = _estimate_ram_power()
        self._stop_event.clear()

        # Detect location if not provided
        if not self._country_iso:
            iso, reg, name, lat, lon = _detect_location()
            self._country_iso = iso
            self._region = reg or self._region
            self._country_name = name
            self._latitude = lat
            self._longitude = lon

        # Cache TDP and carbon intensity at start (don't recompute per measurement)
        self._tdp = _cpu_tdp_fallback()
        self._carbon_intensity = _get_carbon_intensity(self._country_iso, self._region)

        # Start background measurement thread
        self._bg_thread = Thread(target=self._measure_loop, daemon=True)
        self._bg_thread.start()
        return self

    def stop(self) -> float:
        """Stop tracking and return total emissions in kg CO2."""
        self._stop_event.set()
        if self._bg_thread:
            self._bg_thread.join(timeout=5)

        # Final measurement
        self._measure_once()

        duration = time.time() - self._start_time
        total_energy = self._total_cpu_energy + self._total_gpu_energy + self._total_ram_energy
        energy_with_pue = total_energy * self._pue
        carbon_intensity = _get_carbon_intensity(self._country_iso, self._region)
        emissions = energy_with_pue * carbon_intensity
        water = energy_with_pue * self._wue

        data = self._build_data(duration, emissions, total_energy, water)

        if self._save_to_file:
            self._write_csv(data)

        return emissions

    def __enter__(self) -> EmissionsTracker:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Background measurement
    # ------------------------------------------------------------------

    def _measure_loop(self) -> None:
        while not self._stop_event.wait(timeout=self._measure_interval):
            self._measure_once()

    def _measure_once(self) -> None:
        import psutil

        now = time.time()
        dt = now - self._last_measure_time
        if dt <= 0:
            return
        self._last_measure_time = now

        # CPU/GPU power
        if self._is_apple:
            cpu_w, gpu_w = _read_powermetrics(500)
            if cpu_w == 0:
                # Fallback: TDP × utilization
                cpu_pct = psutil.cpu_percent()
                tdp = _cpu_tdp_fallback()
                cpu_w = tdp * (0.1 + 0.9 * (cpu_pct / 100.0) ** 3)
                gpu_w = 0.0
        else:
            cpu_pct = psutil.cpu_percent()
            tdp = _cpu_tdp_fallback()
            cpu_w = tdp * (0.1 + 0.9 * (cpu_pct / 100.0) ** 3)
            gpu_w = 0.0

        self._cpu_power_samples.append(cpu_w)
        self._gpu_power_samples.append(gpu_w)

        # Energy (kWh) = power (W) × time (s) / 3,600,000
        self._total_cpu_energy += cpu_w * dt / 3_600_000
        self._total_gpu_energy += gpu_w * dt / 3_600_000
        self._total_ram_energy += self._ram_power * dt / 3_600_000

        # Utilization samples
        self._cpu_util_samples.append(psutil.cpu_percent())
        vm = psutil.virtual_memory()
        self._ram_util_samples.append(vm.percent)
        self._ram_used_samples.append(vm.used / (1024**3))

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _build_data(
        self,
        duration: float,
        emissions: float,
        energy: float,
        water: float,
    ) -> EmissionsData:
        import psutil

        cpu_model = _get_cpu_model()
        avg = lambda xs: sum(xs) / len(xs) if xs else 0.0

        return EmissionsData(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            project_name=self._project_name,
            run_id=self._run_id,
            experiment_id=self._experiment_id,
            duration=duration,
            emissions=emissions,
            emissions_rate=emissions / duration if duration > 0 else 0,
            cpu_power=avg(self._cpu_power_samples),
            gpu_power=avg(self._gpu_power_samples),
            ram_power=self._ram_power,
            cpu_energy=self._total_cpu_energy,
            gpu_energy=self._total_gpu_energy,
            ram_energy=self._total_ram_energy,
            energy_consumed=energy,
            water_consumed=water,
            country_name=self._country_name,
            country_iso_code=self._country_iso,
            region=self._region,
            os_info=platform.platform(),
            python_version=platform.python_version(),
            tracker_version="morie-1.0.0",
            cpu_count=os.cpu_count() or 0,
            cpu_model=cpu_model,
            gpu_count=1 if self._is_apple else 0,
            gpu_model=cpu_model if self._is_apple else "",
            longitude=self._longitude,
            latitude=self._latitude,
            ram_total_size=psutil.virtual_memory().total / (1024**3),
            tracking_mode=self._tracking_mode,
            cpu_utilization_percent=avg(self._cpu_util_samples),
            gpu_utilization_percent=0.0,
            ram_utilization_percent=avg(self._ram_util_samples),
            ram_used_gb=avg(self._ram_used_samples),
            on_cloud="N",
            pue=self._pue,
            wue=self._wue,
        )

    def _write_csv(self, data: EmissionsData) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / self._output_file
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            if write_header:
                f.write(data.csv_header + "\n")
            f.write(data.csv_row() + "\n")
