Pi Hardware Monitoring
======================

MOIRAIS includes hardware monitoring functions for Raspberry Pi deployments,
enabling Perseus to be self-aware of its host environment. These functions
read system sensors, thermal zones, and device tree data to report real-time
hardware status.

Functions
---------

.. list-table::
   :header-rows: 1
   :widths: 25 45

   * - Function
     - Description
   * - ``pi_status()``
     - Returns a dictionary with CPU temperature, frequency, voltage,
       throttle state, memory usage, disk usage, and uptime
   * - ``pi_health_check()``
     - Boolean health gate: returns ``True`` if temperature < 80C,
       throttle flag is clear, and memory usage < 90%
   * - ``pi_thermal_history()``
     - Returns a list of (timestamp, temperature) tuples from the
       last N readings (default 60, one per minute)
   * - ``format_status()``
     - Formats ``pi_status()`` output as a human-readable multi-line
       string suitable for TUI display or logging

Usage
-----

.. code-block:: python

   from moirais.fn import pi_status, pi_health_check, format_status

   status = pi_status()
   print(f"CPU temp: {status['cpu_temp_c']:.1f} C")
   print(f"Throttled: {status['throttled']}")
   print(f"Memory: {status['mem_used_mb']:.0f}/{status['mem_total_mb']:.0f} MB")

   if not pi_health_check():
       print("WARNING: Pi is overheating or under memory pressure")

   print(format_status(status))

Data Sources
------------

On Raspberry Pi OS, hardware data is read from:

.. list-table::
   :header-rows: 1
   :widths: 35 35

   * - Metric
     - Source
   * - CPU temperature
     - ``/sys/class/thermal/thermal_zone0/temp``
   * - CPU frequency
     - ``/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq``
   * - Voltage / throttle
     - ``vcgencmd measure_volts``, ``vcgencmd get_throttled``
   * - Memory
     - ``/proc/meminfo``
   * - Disk
     - ``os.statvfs()``
   * - Uptime
     - ``/proc/uptime``

On non-Pi platforms (macOS, Linux x86), the functions return ``None`` for
Pi-specific sensors (voltage, throttle) and fall back to standard
``psutil``-style readings where available.

Perseus Self-Awareness
----------------------

When Perseus runs on a Raspberry Pi, it can report its own hardware state
in chat responses. The DoctorScreen (``d`` key in TUI) calls
``pi_health_check()`` as part of its environment diagnostics.

Example chat interaction:

.. code-block:: text

   ? How is the Pi doing?
   Perseus: CPU temperature is 52.3C (normal). Memory usage 43%.
            No throttling detected. Uptime: 3 days, 7 hours.

This is implemented via context injection in ``build_moirais_context()``,
which includes Pi status data when running on ARM64 Linux.

Thermal Management
------------------

The Pi 5 with active cooling (HAT fan) typically runs at 45-55C under
normal LLM inference load. Key thresholds:

- **< 60C**: Normal operation, full clock speed
- **60-80C**: Soft throttle zone, frequency may reduce
- **> 80C**: Hard throttle, ``pi_health_check()`` returns ``False``
- **> 85C**: Emergency shutdown risk

``pi_thermal_history()`` stores readings in a ring buffer, enabling
trend analysis to detect gradual thermal buildup during long inference
sessions.

Integration with Emissions Tracking
------------------------------------

The ``moirais.emissions`` tracker uses Pi sensor data to estimate power
consumption on ARM platforms. CPU temperature and frequency readings
feed into the TDP estimation model, providing per-inference energy
measurements without external power meters.
