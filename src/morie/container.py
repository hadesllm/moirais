"""Docker container management for MORIE.

Provides Python wrappers around Docker and Docker Compose CLI commands for
building, running, inspecting, and managing MORIE analysis containers.  All
functions shell out to the ``docker`` / ``docker compose`` binaries and add
structured output, progress display via ``rich``, and robust error handling.

The container workflow supports:

* Building reproducible analysis images from the current environment.
* Running MORIE modules inside containers with volume-mounted data.
* Inspecting container contents and verifying environment health.
* CI simulation (run the full test suite inside a container).
* Resource monitoring during pipeline execution.
* Ollama sidecar networking for local LLM integration.

References
----------
Docker Engine API:
    https://docs.docker.com/engine/api/

Boettiger, C. (2015). An introduction to Docker for reproducible research.
*ACM SIGOPS Operating Systems Review*, 49(1), 71--79.
https://doi.org/10.1145/2723872.2723882
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MORIE_IMAGE_NAME = "morie"
MORIE_IMAGE_TAG = "latest"
MORIE_CONTAINER_PREFIX = "morie-run"
DEFAULT_WORKDIR = "/morie"
DEFAULT_DATA_MOUNT = "/morie/data"
DEFAULT_OUTPUT_MOUNT = "/morie/output"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DockerResult:
    """Structured result from a Docker command."""

    command: str
    return_code: int
    stdout: str
    stderr: str
    duration_seconds: float

    @property
    def success(self) -> bool:
        """True if the command exited with code 0."""
        return self.return_code == 0


@dataclass
class ContainerInfo:
    """Metadata for a running or stopped MORIE container."""

    container_id: str
    name: str
    image: str
    status: str
    created: str
    ports: str
    size: str = ""


@dataclass
class ImageInfo:
    """Metadata for an MORIE Docker image."""

    image_id: str
    repository: str
    tag: str
    size: str
    created: str


@dataclass
class HealthCheckResult:
    """Result of a container health check."""

    python_ok: bool
    python_version: str
    r_ok: bool
    r_version: str
    morie_importable: bool
    morie_version: str
    data_accessible: bool
    data_path: str
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        """True if all health checks passed."""
        return self.python_ok and self.r_ok and self.morie_importable and self.data_accessible


@dataclass
class ResourceSnapshot:
    """CPU and memory snapshot from a container."""

    timestamp: str
    container_id: str
    cpu_percent: float
    memory_usage_mb: float
    memory_limit_mb: float
    memory_percent: float
    net_io_rx_mb: float
    net_io_tx_mb: float


@dataclass
class EnvironmentDiff:
    """Differences between container and host environments."""

    host_python: str
    container_python: str
    host_r: str
    container_r: str
    host_os: str
    container_os: str
    package_diffs: list[dict[str, str]] = field(default_factory=list)
    missing_on_host: list[str] = field(default_factory=list)
    missing_in_container: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Low-level runner
# ---------------------------------------------------------------------------


def _run_docker(
    args: list[str],
    *,
    timeout: int = 600,
    capture: bool = True,
    env: dict[str, str] | None = None,
) -> DockerResult:
    """Execute a docker command and return structured output.

    Parameters
    ----------
    args : list[str]
        Command arguments (without the leading ``docker``).
    timeout : int, optional
        Maximum seconds to wait (default 600).
    capture : bool, optional
        If True, capture stdout/stderr; otherwise stream to terminal.
    env : dict[str, str] | None, optional
        Extra environment variables to pass.

    Returns
    -------
    DockerResult
        Structured command result.

    Raises
    ------
    FileNotFoundError
        If the ``docker`` binary is not found on PATH.
    """
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        raise FileNotFoundError(
            "Docker CLI not found on PATH. Install Docker Desktop or Docker Engine: https://docs.docker.com/get-docker/"
        )

    cmd = [docker_bin] + args
    logger.debug("Running: %s", " ".join(cmd))

    run_env = {**os.environ, **(env or {})}
    start = time.monotonic()

    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout,
            env=run_env,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return DockerResult(
            command=" ".join(cmd),
            return_code=-1,
            stdout="",
            stderr=f"Command timed out after {timeout}s",
            duration_seconds=elapsed,
        )

    elapsed = time.monotonic() - start
    return DockerResult(
        command=" ".join(cmd),
        return_code=result.returncode,
        stdout=result.stdout if capture else "",
        stderr=result.stderr if capture else "",
        duration_seconds=elapsed,
    )


def _run_compose(
    args: list[str],
    *,
    compose_file: str | Path | None = None,
    timeout: int = 600,
) -> DockerResult:
    """Execute a docker compose command.

    Parameters
    ----------
    args : list[str]
        Compose sub-command arguments.
    compose_file : str | Path | None, optional
        Path to compose YAML. If None, uses default discovery.
    timeout : int, optional
        Maximum seconds to wait (default 600).

    Returns
    -------
    DockerResult
    """
    cmd = ["compose"]
    if compose_file is not None:
        cmd.extend(["-f", str(compose_file)])
    cmd.extend(args)
    return _run_docker(cmd, timeout=timeout)


def docker_available() -> bool:
    """Check whether the Docker CLI is accessible and the daemon is running.

    Returns
    -------
    bool
        True if ``docker info`` succeeds.
    """
    try:
        result = _run_docker(["info"], timeout=15)
        return result.success
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Image management
# ---------------------------------------------------------------------------


def build_image(
    context_dir: str | Path = ".",
    *,
    image_name: str = MORIE_IMAGE_NAME,
    tag: str = MORIE_IMAGE_TAG,
    dockerfile: str | Path | None = None,
    build_args: dict[str, str] | None = None,
    no_cache: bool = False,
    target: str | None = None,
    platform_arch: str | None = None,
    timeout: int = 1200,
) -> DockerResult:
    """Build the MORIE Docker image.

    Wraps ``docker build`` with sensible defaults for the MORIE project.

    Parameters
    ----------
    context_dir : str | Path
        Build context directory (default: current directory).
    image_name : str
        Image repository name (default ``morie``).
    tag : str
        Image tag (default ``latest``).
    dockerfile : str | Path | None
        Path to Dockerfile (default: ``context_dir/Dockerfile``).
    build_args : dict[str, str] | None
        ``--build-arg`` key=value pairs.
    no_cache : bool
        If True, pass ``--no-cache``.
    target : str | None
        Multi-stage build target stage name.
    platform_arch : str | None
        Target platform (e.g. ``linux/amd64``).
    timeout : int
        Build timeout in seconds (default 1200).

    Returns
    -------
    DockerResult
    """
    full_tag = f"{image_name}:{tag}"
    cmd = ["build", "-t", full_tag]

    if dockerfile is not None:
        cmd.extend(["-f", str(dockerfile)])
    if no_cache:
        cmd.append("--no-cache")
    if target is not None:
        cmd.extend(["--target", target])
    if platform_arch is not None:
        cmd.extend(["--platform", platform_arch])
    if build_args:
        for k, v in build_args.items():
            cmd.extend(["--build-arg", f"{k}={v}"])

    cmd.append(str(context_dir))

    logger.info("Building image %s from %s", full_tag, context_dir)
    return _run_docker(cmd, timeout=timeout, capture=False)


def build_multistage(
    context_dir: str | Path = ".",
    *,
    image_name: str = MORIE_IMAGE_NAME,
    stages: list[str] | None = None,
    final_tag: str = MORIE_IMAGE_TAG,
    timeout: int = 1800,
) -> list[DockerResult]:
    """Orchestrate a multi-stage Docker build.

    Builds each named stage separately (useful for caching and debugging)
    and then builds the final image.

    Parameters
    ----------
    context_dir : str | Path
        Build context.
    image_name : str
        Base image name.
    stages : list[str] | None
        Stage names to build individually before the final build.
        If None, only the final build is executed.
    final_tag : str
        Tag for the final image.
    timeout : int
        Per-stage timeout in seconds.

    Returns
    -------
    list[DockerResult]
        One result per stage plus the final build.
    """
    results: list[DockerResult] = []
    if stages:
        for stage in stages:
            stage_tag = f"{image_name}:{stage}"
            r = build_image(
                context_dir,
                image_name=image_name,
                tag=stage,
                target=stage,
                timeout=timeout,
            )
            results.append(r)
            if not r.success:
                logger.error("Stage '%s' failed", stage)
                return results

    final = build_image(context_dir, image_name=image_name, tag=final_tag, timeout=timeout)
    results.append(final)
    return results


def list_images(
    *,
    filter_morie: bool = True,
) -> list[ImageInfo]:
    """List Docker images, optionally filtered to MORIE images.

    Parameters
    ----------
    filter_morie : bool
        If True, only return images with repository matching MORIE_IMAGE_NAME.

    Returns
    -------
    list[ImageInfo]
    """
    fmt = "{{.ID}}\t{{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    cmd = ["images", f"--format={fmt}"]
    if filter_morie:
        cmd.append(MORIE_IMAGE_NAME)

    result = _run_docker(cmd)
    images: list[ImageInfo] = []
    if not result.success:
        logger.warning("Failed to list images: %s", result.stderr)
        return images

    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 5:
            images.append(
                ImageInfo(
                    image_id=parts[0],
                    repository=parts[1],
                    tag=parts[2],
                    size=parts[3],
                    created=parts[4],
                )
            )
    return images


def remove_image(
    image: str,
    *,
    force: bool = False,
) -> DockerResult:
    """Remove a Docker image.

    Parameters
    ----------
    image : str
        Image name or ID.
    force : bool
        If True, force removal.

    Returns
    -------
    DockerResult
    """
    cmd = ["rmi"]
    if force:
        cmd.append("--force")
    cmd.append(image)
    return _run_docker(cmd)


def cleanup_images(
    *,
    keep_latest: bool = True,
    max_age_days: int = 30,
) -> list[DockerResult]:
    """Remove old MORIE images.

    Parameters
    ----------
    keep_latest : bool
        If True, keep the ``latest`` tagged image.
    max_age_days : int
        Remove images older than this many days.

    Returns
    -------
    list[DockerResult]
        One result per removed image.
    """
    images = list_images(filter_morie=True)
    results: list[DockerResult] = []

    for img in images:
        if keep_latest and img.tag == "latest":
            continue
        # Parse creation date; docker format varies, try common patterns
        try:
            created_str = img.created.split(" +")[0].strip()
            created_dt = datetime.strptime(created_str, "%Y-%m-%d %H:%M:%S")
            age = (datetime.now() - created_dt).days
            if age <= max_age_days:
                continue
        except (ValueError, IndexError):
            # Cannot parse date; skip to be safe
            continue

        r = remove_image(f"{img.repository}:{img.tag}", force=True)
        results.append(r)
        if r.success:
            logger.info("Removed image %s:%s (age %d days)", img.repository, img.tag, age)

    return results


def push_image(
    image: str = f"{MORIE_IMAGE_NAME}:{MORIE_IMAGE_TAG}",
    *,
    registry: str | None = None,
) -> DockerResult:
    """Push an image to a registry.

    Parameters
    ----------
    image : str
        Local image tag.
    registry : str | None
        Registry prefix (e.g. ``ghcr.io/hadesllm``). If provided, the image
        is tagged with the registry prefix before pushing.

    Returns
    -------
    DockerResult
    """
    target = image
    if registry:
        target = f"{registry}/{image}"
        tag_result = _run_docker(["tag", image, target])
        if not tag_result.success:
            return tag_result

    return _run_docker(["push", target], timeout=600)


def pull_image(
    image: str = f"{MORIE_IMAGE_NAME}:{MORIE_IMAGE_TAG}",
    *,
    registry: str | None = None,
) -> DockerResult:
    """Pull an image from a registry.

    Parameters
    ----------
    image : str
        Image tag.
    registry : str | None
        Registry prefix.

    Returns
    -------
    DockerResult
    """
    target = f"{registry}/{image}" if registry else image
    return _run_docker(["pull", target], timeout=600)


# ---------------------------------------------------------------------------
# Container lifecycle
# ---------------------------------------------------------------------------


def run_container(
    *,
    image: str = f"{MORIE_IMAGE_NAME}:{MORIE_IMAGE_TAG}",
    name: str | None = None,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    config_dir: str | Path | None = None,
    command: str | None = None,
    env_vars: dict[str, str] | None = None,
    ports: dict[int, int] | None = None,
    detach: bool = False,
    remove: bool = True,
    network: str | None = None,
    cpus: float | None = None,
    memory: str | None = None,
    timeout: int = 3600,
) -> DockerResult:
    """Run an MORIE container.

    Parameters
    ----------
    image : str
        Docker image to run.
    name : str | None
        Container name. Auto-generated if None.
    data_dir : str | Path | None
        Host directory to mount at ``/morie/data``.
    output_dir : str | Path | None
        Host directory to mount at ``/morie/output``.
    config_dir : str | Path | None
        Host directory to mount at ``/morie/config``.
    command : str | None
        Override the default CMD.
    env_vars : dict[str, str] | None
        Environment variables to set in the container.
    ports : dict[int, int] | None
        Port mappings ``{host_port: container_port}``.
    detach : bool
        If True, run in background.
    remove : bool
        If True, auto-remove container on exit.
    network : str | None
        Docker network to attach to.
    cpus : float | None
        CPU limit (e.g. 2.0 for two cores).
    memory : str | None
        Memory limit (e.g. ``4g``).
    timeout : int
        Timeout in seconds (default 3600).

    Returns
    -------
    DockerResult
    """
    if name is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"{MORIE_CONTAINER_PREFIX}-{ts}"

    cmd = ["run", "--name", name]

    if detach:
        cmd.append("-d")
    if remove and not detach:
        cmd.append("--rm")

    # Volume mounts
    if data_dir is not None:
        data_path = Path(data_dir).resolve()
        cmd.extend(["-v", f"{data_path}:{DEFAULT_DATA_MOUNT}:ro"])
    if output_dir is not None:
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        cmd.extend(["-v", f"{output_path}:{DEFAULT_OUTPUT_MOUNT}"])
    if config_dir is not None:
        config_path = Path(config_dir).resolve()
        cmd.extend(["-v", f"{config_path}:/morie/config:ro"])

    # Environment variables
    if env_vars:
        for k, v in env_vars.items():
            cmd.extend(["-e", f"{k}={v}"])

    # Port mappings
    if ports:
        for host_port, container_port in ports.items():
            cmd.extend(["-p", f"{host_port}:{container_port}"])

    # Network
    if network:
        cmd.extend(["--network", network])

    # Resource limits
    if cpus is not None:
        cmd.extend(["--cpus", str(cpus)])
    if memory is not None:
        cmd.extend(["--memory", memory])

    cmd.append(image)

    if command:
        cmd.extend(command.split())

    logger.info("Starting container %s from %s", name, image)
    return _run_docker(cmd, timeout=timeout, capture=not detach)


def exec_in_container(
    container: str,
    command: str,
    *,
    workdir: str | None = None,
    user: str | None = None,
    env_vars: dict[str, str] | None = None,
    interactive: bool = False,
    timeout: int = 300,
) -> DockerResult:
    """Execute a command inside a running container.

    Parameters
    ----------
    container : str
        Container name or ID.
    command : str
        Command to execute.
    workdir : str | None
        Working directory inside the container.
    user : str | None
        User to run as (e.g. ``root``).
    env_vars : dict[str, str] | None
        Additional environment variables.
    interactive : bool
        If True, pass ``-it`` flags.
    timeout : int
        Timeout in seconds.

    Returns
    -------
    DockerResult
    """
    cmd = ["exec"]
    if interactive:
        cmd.extend(["-it"])
    if workdir:
        cmd.extend(["-w", workdir])
    if user:
        cmd.extend(["-u", user])
    if env_vars:
        for k, v in env_vars.items():
            cmd.extend(["-e", f"{k}={v}"])

    cmd.append(container)
    cmd.extend(command.split())

    return _run_docker(cmd, timeout=timeout)


def container_shell(
    container: str,
    *,
    shell: str = "/bin/bash",
) -> DockerResult:
    """Open an interactive shell in a running container.

    Parameters
    ----------
    container : str
        Container name or ID.
    shell : str
        Shell binary (default ``/bin/bash``).

    Returns
    -------
    DockerResult
    """
    return exec_in_container(container, shell, interactive=True)


def stop_container(
    container: str,
    *,
    timeout_seconds: int = 10,
) -> DockerResult:
    """Stop a running container gracefully.

    Parameters
    ----------
    container : str
        Container name or ID.
    timeout_seconds : int
        Grace period before SIGKILL.

    Returns
    -------
    DockerResult
    """
    return _run_docker(["stop", "-t", str(timeout_seconds), container])


def remove_container(
    container: str,
    *,
    force: bool = False,
    remove_volumes: bool = False,
) -> DockerResult:
    """Remove a container.

    Parameters
    ----------
    container : str
        Container name or ID.
    force : bool
        If True, force removal of running container.
    remove_volumes : bool
        If True, also remove associated volumes.

    Returns
    -------
    DockerResult
    """
    cmd = ["rm"]
    if force:
        cmd.append("--force")
    if remove_volumes:
        cmd.append("-v")
    cmd.append(container)
    return _run_docker(cmd)


def list_containers(
    *,
    all_states: bool = True,
    filter_morie: bool = True,
) -> list[ContainerInfo]:
    """List MORIE containers.

    Parameters
    ----------
    all_states : bool
        Include stopped containers.
    filter_morie : bool
        Only show containers with names starting with the MORIE prefix.

    Returns
    -------
    list[ContainerInfo]
    """
    fmt = "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.CreatedAt}}\t{{.Ports}}\t{{.Size}}"
    cmd = ["ps", f"--format={fmt}", "--size"]
    if all_states:
        cmd.append("-a")
    if filter_morie:
        cmd.extend(["--filter", f"name={MORIE_CONTAINER_PREFIX}"])

    result = _run_docker(cmd)
    containers: list[ContainerInfo] = []
    if not result.success:
        return containers

    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 6:
            containers.append(
                ContainerInfo(
                    container_id=parts[0],
                    name=parts[1],
                    image=parts[2],
                    status=parts[3],
                    created=parts[4],
                    ports=parts[5],
                    size=parts[6] if len(parts) > 6 else "",
                )
            )
    return containers


def cleanup_containers(
    *,
    remove_stopped: bool = True,
    remove_running: bool = False,
) -> list[DockerResult]:
    """Remove old MORIE containers.

    Parameters
    ----------
    remove_stopped : bool
        Remove containers that have exited.
    remove_running : bool
        Force-remove running containers.

    Returns
    -------
    list[DockerResult]
    """
    containers = list_containers(all_states=True, filter_morie=True)
    results: list[DockerResult] = []

    for c in containers:
        is_running = "Up" in c.status
        if is_running and not remove_running:
            continue
        if not is_running and not remove_stopped:
            continue
        r = remove_container(c.container_id, force=is_running)
        results.append(r)
        if r.success:
            logger.info("Removed container %s (%s)", c.name, c.container_id[:12])

    return results


# ---------------------------------------------------------------------------
# Inspection and health checks
# ---------------------------------------------------------------------------


def inspect_container(
    container: str,
    *,
    list_files: bool = True,
    check_versions: bool = True,
) -> dict[str, Any]:
    """Inspect a container: list files, check installed versions.

    Parameters
    ----------
    container : str
        Container name or ID.
    list_files : bool
        List files in the MORIE working directory.
    check_versions : bool
        Check Python, R, and key package versions.

    Returns
    -------
    dict[str, Any]
        Inspection results with keys ``files``, ``python_version``,
        ``r_version``, ``packages``.
    """
    info: dict[str, Any] = {"container": container}

    if list_files:
        r = exec_in_container(container, f"find {DEFAULT_WORKDIR} -type f -maxdepth 3")
        info["files"] = r.stdout.strip().splitlines() if r.success else []

    if check_versions:
        py = exec_in_container(container, "python --version")
        info["python_version"] = py.stdout.strip() if py.success else "not found"

        rv = exec_in_container(container, "R --version")
        if rv.success:
            first_line = rv.stdout.strip().splitlines()[0] if rv.stdout.strip() else ""
            info["r_version"] = first_line
        else:
            info["r_version"] = "not found"

        pip_list = exec_in_container(container, "pip list --format=json")
        if pip_list.success:
            try:
                info["packages"] = json.loads(pip_list.stdout)
            except json.JSONDecodeError:
                info["packages"] = []
        else:
            info["packages"] = []

    return info


def health_check(
    container: str,
    *,
    data_path: str = DEFAULT_DATA_MOUNT,
) -> HealthCheckResult:
    """Run health checks on a container.

    Verifies that Python, R, and the ``morie`` package are importable and
    that the data directory is accessible.

    Parameters
    ----------
    container : str
        Container name or ID.
    data_path : str
        Expected data mount path inside the container.

    Returns
    -------
    HealthCheckResult
    """
    checks: list[dict[str, Any]] = []

    # Python
    py = exec_in_container(container, 'python -c "import sys; print(sys.version)"')
    python_ok = py.success
    python_version = py.stdout.strip() if py.success else py.stderr.strip()
    checks.append({"name": "python", "ok": python_ok, "detail": python_version})

    # R
    rv = exec_in_container(container, 'R --slave -e "cat(R.version.string)"')
    r_ok = rv.success
    r_version = rv.stdout.strip() if rv.success else "not available"
    checks.append({"name": "r", "ok": r_ok, "detail": r_version})

    # morie importable
    morie_check = exec_in_container(
        container,
        'python -c "import morie; print(morie.__version__)"',
    )
    morie_importable = morie_check.success
    morie_version = morie_check.stdout.strip() if morie_check.success else "not importable"
    checks.append({"name": "morie", "ok": morie_importable, "detail": morie_version})

    # Data directory
    data_check = exec_in_container(container, f"test -d {data_path} && echo ok")
    data_accessible = data_check.success and "ok" in data_check.stdout
    checks.append({"name": "data_dir", "ok": data_accessible, "detail": data_path})

    return HealthCheckResult(
        python_ok=python_ok,
        python_version=python_version,
        r_ok=r_ok,
        r_version=r_version,
        morie_importable=morie_importable,
        morie_version=morie_version,
        data_accessible=data_accessible,
        data_path=data_path,
        checks=checks,
    )


def environment_diff(
    container: str,
) -> EnvironmentDiff:
    """Compare container environment against the host.

    Checks Python version, R version, OS, and installed Python packages
    on both host and container, then reports differences.

    Parameters
    ----------
    container : str
        Container name or ID.

    Returns
    -------
    EnvironmentDiff
    """
    import importlib.metadata

    # Host info
    host_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    host_os = f"{platform.system()} {platform.release()}"

    r_result = subprocess.run(
        ["R", "--slave", "-e", "cat(R.version.string)"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    host_r = r_result.stdout.strip() if r_result.returncode == 0 else "not available"

    host_packages: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        version = dist.metadata["Version"]
        if name:
            host_packages[name.lower()] = version

    # Container info
    py = exec_in_container(
        container,
        "python -c \"import sys; v=sys.version_info; print(f'{v.major}.{v.minor}.{v.micro}')\"",
    )
    container_python = py.stdout.strip() if py.success else "unknown"

    rv = exec_in_container(container, 'R --slave -e "cat(R.version.string)"')
    container_r = rv.stdout.strip() if rv.success else "not available"

    os_check = exec_in_container(container, "cat /etc/os-release")
    container_os = "unknown"
    if os_check.success:
        for line in os_check.stdout.splitlines():
            if line.startswith("PRETTY_NAME="):
                container_os = line.split("=", 1)[1].strip('"')
                break

    pip_json = exec_in_container(container, "pip list --format=json")
    container_packages: dict[str, str] = {}
    if pip_json.success:
        try:
            for pkg in json.loads(pip_json.stdout):
                container_packages[pkg["name"].lower()] = pkg["version"]
        except (json.JSONDecodeError, KeyError):
            pass

    # Compute diffs
    all_pkgs = set(host_packages.keys()) | set(container_packages.keys())
    package_diffs: list[dict[str, str]] = []
    missing_on_host: list[str] = []
    missing_in_container: list[str] = []

    for pkg in sorted(all_pkgs):
        h_ver = host_packages.get(pkg)
        c_ver = container_packages.get(pkg)
        if h_ver and c_ver and h_ver != c_ver:
            package_diffs.append({"package": pkg, "host": h_ver, "container": c_ver})
        elif h_ver and not c_ver:
            missing_in_container.append(pkg)
        elif c_ver and not h_ver:
            missing_on_host.append(pkg)

    return EnvironmentDiff(
        host_python=host_python,
        container_python=container_python,
        host_r=host_r,
        container_r=container_r,
        host_os=host_os,
        container_os=container_os,
        package_diffs=package_diffs,
        missing_on_host=missing_on_host,
        missing_in_container=missing_in_container,
    )


# ---------------------------------------------------------------------------
# Pipeline and verification
# ---------------------------------------------------------------------------


def verify_pipeline(
    container: str,
    *,
    modules: list[str] | None = None,
    cpads_csv: str | None = None,
    output_dir: str = DEFAULT_OUTPUT_MOUNT,
    timeout: int = 3600,
) -> dict[str, Any]:
    """Run MORIE modules inside a container and verify outputs.

    Parameters
    ----------
    container : str
        Container name or ID.
    modules : list[str] | None
        Module names to run. If None, runs all.
    cpads_csv : str | None
        Path to CPADS CSV inside the container.
    output_dir : str
        Output directory inside the container.
    timeout : int
        Per-module timeout in seconds.

    Returns
    -------
    dict[str, Any]
        Keys: ``results`` (per-module status), ``all_passed`` (bool),
        ``duration_seconds`` (float).
    """
    start = time.monotonic()
    results: list[dict[str, Any]] = []

    # Get available modules
    if modules is None:
        mod_list = exec_in_container(
            container,
            "python -m morie.runner list-modules",
        )
        if mod_list.success:
            modules = [
                line.strip().lstrip("- ").split()[0]
                for line in mod_list.stdout.strip().splitlines()
                if line.strip().startswith("-") or line.strip()
            ]
        else:
            modules = []

    for mod in modules:
        cmd_parts = ["python", "-m", "morie.runner", "run-module", mod, "--output-dir", output_dir]
        if cpads_csv:
            cmd_parts.extend(["--cpads-csv", cpads_csv])
        cmd_str = " ".join(cmd_parts)

        r = exec_in_container(container, cmd_str, timeout=timeout)
        results.append(
            {
                "module": mod,
                "success": r.success,
                "return_code": r.return_code,
                "duration": r.duration_seconds,
                "stderr": r.stderr[:500] if not r.success else "",
            }
        )
        logger.info("Module %s: %s (%.1fs)", mod, "OK" if r.success else "FAILED", r.duration_seconds)

    elapsed = time.monotonic() - start
    return {
        "results": results,
        "all_passed": all(r["success"] for r in results),
        "duration_seconds": elapsed,
    }


def run_ci_simulation(
    *,
    image: str = f"{MORIE_IMAGE_NAME}:{MORIE_IMAGE_TAG}",
    test_command: str = "python -m pytest -q --tb=short",
    data_dir: str | Path | None = None,
    timeout: int = 1800,
) -> DockerResult:
    """Simulate CI by running the test suite inside a container.

    Parameters
    ----------
    image : str
        Docker image to use.
    test_command : str
        Test command to execute.
    data_dir : str | Path | None
        Host data directory to mount.
    timeout : int
        Test timeout in seconds.

    Returns
    -------
    DockerResult
    """
    return run_container(
        image=image,
        name=f"morie-ci-{int(time.time())}",
        data_dir=data_dir,
        command=test_command,
        remove=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Data import / export
# ---------------------------------------------------------------------------


def export_results(
    container: str,
    container_path: str,
    host_path: str | Path,
) -> DockerResult:
    """Copy files from a container to the host.

    Parameters
    ----------
    container : str
        Container name or ID.
    container_path : str
        Source path inside the container.
    host_path : str | Path
        Destination path on the host.

    Returns
    -------
    DockerResult
    """
    host_path = Path(host_path)
    host_path.parent.mkdir(parents=True, exist_ok=True)
    return _run_docker(["cp", f"{container}:{container_path}", str(host_path)])


def import_data(
    container: str,
    host_path: str | Path,
    container_path: str = DEFAULT_DATA_MOUNT,
) -> DockerResult:
    """Copy data files from the host into a container.

    Parameters
    ----------
    container : str
        Container name or ID.
    host_path : str | Path
        Source path on the host.
    container_path : str
        Destination path inside the container.

    Returns
    -------
    DockerResult
    """
    host_path = Path(host_path)
    if not host_path.exists():
        raise FileNotFoundError(f"Host path not found: {host_path}")
    return _run_docker(["cp", str(host_path), f"{container}:{container_path}"])


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def snapshot_container(
    container: str,
    *,
    tag: str | None = None,
    message: str | None = None,
) -> DockerResult:
    """Commit the current state of a container as a new image.

    Parameters
    ----------
    container : str
        Container name or ID.
    tag : str | None
        Tag for the new image. Auto-generated if None.
    message : str | None
        Commit message.

    Returns
    -------
    DockerResult
    """
    if tag is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        tag = f"{MORIE_IMAGE_NAME}:snapshot-{ts}"
    else:
        if ":" not in tag:
            tag = f"{MORIE_IMAGE_NAME}:{tag}"

    cmd = ["commit"]
    if message:
        cmd.extend(["-m", message])
    cmd.extend([container, tag])
    return _run_docker(cmd)


# ---------------------------------------------------------------------------
# Resource monitoring
# ---------------------------------------------------------------------------


def get_resource_stats(
    container: str,
) -> ResourceSnapshot | None:
    """Get a single resource usage snapshot for a container.

    Parameters
    ----------
    container : str
        Container name or ID.

    Returns
    -------
    ResourceSnapshot | None
        Snapshot of CPU, memory, and network usage.  None if the container
        is not running or stats cannot be read.
    """
    fmt = "{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}"
    result = _run_docker(["stats", "--no-stream", f"--format={fmt}", container])
    if not result.success:
        return None

    line = result.stdout.strip()
    if not line:
        return None

    parts = line.split("\t")
    if len(parts) < 4:
        return None

    def _parse_percent(s: str) -> float:
        return float(s.strip().rstrip("%"))

    def _parse_mem(s: str) -> tuple[float, float]:
        # "123.4MiB / 7.773GiB"
        chunks = s.split("/")
        usage = _parse_size_mb(chunks[0].strip()) if len(chunks) > 0 else 0.0
        limit = _parse_size_mb(chunks[1].strip()) if len(chunks) > 1 else 0.0
        return usage, limit

    def _parse_size_mb(s: str) -> float:
        s = s.strip()
        if s.endswith("GiB"):
            return float(s[:-3]) * 1024
        if s.endswith("MiB"):
            return float(s[:-3])
        if s.endswith("KiB"):
            return float(s[:-3]) / 1024
        if s.endswith("B"):
            return float(s[:-1]) / (1024 * 1024)
        return 0.0

    def _parse_net(s: str) -> tuple[float, float]:
        chunks = s.split("/")
        rx = _parse_size_mb(chunks[0].strip()) if len(chunks) > 0 else 0.0
        tx = _parse_size_mb(chunks[1].strip()) if len(chunks) > 1 else 0.0
        return rx, tx

    try:
        cpu = _parse_percent(parts[0])
        mem_usage, mem_limit = _parse_mem(parts[1])
        mem_pct = _parse_percent(parts[2])
        net_rx, net_tx = _parse_net(parts[3])
    except (ValueError, IndexError):
        return None

    return ResourceSnapshot(
        timestamp=datetime.now(timezone.utc).isoformat(),
        container_id=container,
        cpu_percent=cpu,
        memory_usage_mb=mem_usage,
        memory_limit_mb=mem_limit,
        memory_percent=mem_pct,
        net_io_rx_mb=net_rx,
        net_io_tx_mb=net_tx,
    )


def monitor_resources(
    container: str,
    *,
    duration_seconds: int = 60,
    interval_seconds: float = 2.0,
) -> list[ResourceSnapshot]:
    """Monitor container resource usage over time.

    Parameters
    ----------
    container : str
        Container name or ID.
    duration_seconds : int
        Total monitoring duration.
    interval_seconds : float
        Seconds between samples.

    Returns
    -------
    list[ResourceSnapshot]
        Time series of resource snapshots.
    """
    snapshots: list[ResourceSnapshot] = []
    end_time = time.monotonic() + duration_seconds

    while time.monotonic() < end_time:
        snap = get_resource_stats(container)
        if snap is not None:
            snapshots.append(snap)
        time.sleep(interval_seconds)

    logger.info("Collected %d resource snapshots over %ds", len(snapshots), duration_seconds)
    return snapshots


# ---------------------------------------------------------------------------
# Container logs
# ---------------------------------------------------------------------------


def get_container_logs(
    container: str,
    *,
    tail: int | None = None,
    since: str | None = None,
    follow: bool = False,
    timestamps: bool = True,
) -> DockerResult:
    """Retrieve logs from a container.

    Parameters
    ----------
    container : str
        Container name or ID.
    tail : int | None
        Only return the last N lines.
    since : str | None
        Show logs since a timestamp or duration (e.g. ``10m``, ``2024-01-01``).
    follow : bool
        If True, follow log output (blocking).
    timestamps : bool
        If True, include timestamps.

    Returns
    -------
    DockerResult
    """
    cmd = ["logs"]
    if tail is not None:
        cmd.extend(["--tail", str(tail)])
    if since:
        cmd.extend(["--since", since])
    if follow:
        cmd.append("--follow")
    if timestamps:
        cmd.append("--timestamps")
    cmd.append(container)

    return _run_docker(cmd, timeout=30 if not follow else 3600)


def render_container_logs(
    container: str,
    *,
    tail: int = 100,
) -> None:
    """Display container logs with rich formatting.

    Parameters
    ----------
    container : str
        Container name or ID.
    tail : int
        Number of lines to display.
    """
    result = get_container_logs(container, tail=tail)
    if not result.success:
        logger.error("Failed to get logs: %s", result.stderr)
        return

    try:
        from rich.console import Console
        from rich.syntax import Syntax

        console = Console()
        console.print(f"\n[bold cyan]Logs: {container}[/bold cyan] (last {tail} lines)\n")
        console.print(Syntax(result.stdout, "log", theme="monokai"))
    except ImportError:
        print(f"Logs: {container} (last {tail} lines)")
        print(result.stdout)


# ---------------------------------------------------------------------------
# Docker Compose management
# ---------------------------------------------------------------------------


def compose_up(
    *,
    compose_file: str | Path | None = None,
    services: list[str] | None = None,
    detach: bool = True,
    build: bool = False,
    timeout: int = 300,
) -> DockerResult:
    """Start services defined in docker-compose.yml.

    Parameters
    ----------
    compose_file : str | Path | None
        Compose file path.
    services : list[str] | None
        Specific services to start. If None, starts all.
    detach : bool
        Run in background.
    build : bool
        Rebuild images before starting.
    timeout : int
        Startup timeout.

    Returns
    -------
    DockerResult
    """
    cmd = ["up"]
    if detach:
        cmd.append("-d")
    if build:
        cmd.append("--build")
    if services:
        cmd.extend(services)
    return _run_compose(cmd, compose_file=compose_file, timeout=timeout)


def compose_down(
    *,
    compose_file: str | Path | None = None,
    remove_volumes: bool = False,
    remove_images: str | None = None,
    timeout: int = 120,
) -> DockerResult:
    """Stop and remove compose services.

    Parameters
    ----------
    compose_file : str | Path | None
        Compose file path.
    remove_volumes : bool
        Remove named volumes.
    remove_images : str | None
        ``all`` or ``local`` to remove images.
    timeout : int
        Shutdown timeout.

    Returns
    -------
    DockerResult
    """
    cmd = ["down"]
    if remove_volumes:
        cmd.append("-v")
    if remove_images:
        cmd.extend(["--rmi", remove_images])
    return _run_compose(cmd, compose_file=compose_file, timeout=timeout)


def compose_ps(
    *,
    compose_file: str | Path | None = None,
) -> DockerResult:
    """List running compose services.

    Parameters
    ----------
    compose_file : str | Path | None
        Compose file path.

    Returns
    -------
    DockerResult
    """
    return _run_compose(["ps"], compose_file=compose_file)


def compose_logs(
    *,
    compose_file: str | Path | None = None,
    services: list[str] | None = None,
    tail: int = 100,
) -> DockerResult:
    """Retrieve logs from compose services.

    Parameters
    ----------
    compose_file : str | Path | None
        Compose file path.
    services : list[str] | None
        Specific services.
    tail : int
        Number of lines per service.

    Returns
    -------
    DockerResult
    """
    cmd = ["logs", "--tail", str(tail)]
    if services:
        cmd.extend(services)
    return _run_compose(cmd, compose_file=compose_file)


# ---------------------------------------------------------------------------
# Network configuration (Ollama sidecar)
# ---------------------------------------------------------------------------


def create_network(
    name: str = "morie-net",
    *,
    driver: str = "bridge",
) -> DockerResult:
    """Create a Docker network for MORIE services.

    Parameters
    ----------
    name : str
        Network name.
    driver : str
        Network driver (default ``bridge``).

    Returns
    -------
    DockerResult
    """
    return _run_docker(["network", "create", "--driver", driver, name])


def remove_network(name: str = "morie-net") -> DockerResult:
    """Remove a Docker network.

    Parameters
    ----------
    name : str
        Network name.

    Returns
    -------
    DockerResult
    """
    return _run_docker(["network", "rm", name])


def setup_ollama_sidecar(
    *,
    network: str = "morie-net",
    ollama_image: str = "ollama/ollama:latest",
    model: str = "qwen2.5:7b",
    gpu: bool = False,
) -> dict[str, DockerResult]:
    """Set up an Ollama sidecar container for local LLM inference.

    Creates a Docker network, starts an Ollama container, and pulls the
    specified model.

    Parameters
    ----------
    network : str
        Docker network name.
    ollama_image : str
        Ollama Docker image.
    model : str
        Model to pull (e.g. ``qwen2.5:7b``).
    gpu : bool
        If True, pass ``--gpus=all`` to enable GPU acceleration.

    Returns
    -------
    dict[str, DockerResult]
        Keys: ``network``, ``container``, ``model_pull``.
    """
    results: dict[str, DockerResult] = {}

    # Create network (ignore error if already exists)
    net = create_network(network)
    results["network"] = net

    # Start Ollama container
    cmd = [
        "run",
        "-d",
        "--name",
        "morie-ollama",
        "--network",
        network,
        "-p",
        "11434:11434",
    ]
    if gpu:
        cmd.extend(["--gpus", "all"])
    cmd.append(ollama_image)

    results["container"] = _run_docker(cmd)

    if results["container"].success:
        # Wait for Ollama to start
        time.sleep(3)
        # Pull model
        results["model_pull"] = exec_in_container(
            "morie-ollama",
            f"ollama pull {model}",
            timeout=600,
        )
    else:
        results["model_pull"] = DockerResult(
            command="skipped",
            return_code=-1,
            stdout="",
            stderr="Container did not start",
            duration_seconds=0,
        )

    return results


# ---------------------------------------------------------------------------
# Dockerfile generation
# ---------------------------------------------------------------------------


def generate_dockerfile(
    *,
    python_version: str = "3.11",
    include_r: bool = True,
    include_quarto: bool = False,
    extras: list[str] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """Generate a Dockerfile for the current MORIE environment.

    Creates a multi-stage Dockerfile that installs Python, optionally R
    and Quarto, and the morie package with requested extras.

    Parameters
    ----------
    python_version : str
        Python version for the base image.
    include_r : bool
        Include R installation.
    include_quarto : bool
        Include Quarto CLI installation.
    extras : list[str] | None
        pip extras to install (e.g. ``["ai", "carbon"]``).
    output_path : str | Path | None
        If provided, write the Dockerfile to this path.

    Returns
    -------
    str
        Dockerfile contents.
    """
    extras_str = ""
    if extras:
        extras_str = "[" + ",".join(extras) + "]"

    r_block = ""
    if include_r:
        r_block = textwrap.dedent("""\
            # --- R installation ---
            RUN apt-get update && apt-get install -y --no-install-recommends \\
                r-base r-base-dev \\
                libcurl4-openssl-dev libssl-dev libxml2-dev \\
                && rm -rf /var/lib/apt/lists/*

            # Install R packages
            RUN R -e "install.packages(c('testthat', 'roxygen2', 'devtools', 'survey', 'MatchIt', 'WeightIt'), repos='https://cloud.r-project.org')"
        """)

    quarto_block = ""
    if include_quarto:
        quarto_block = textwrap.dedent("""\
            # --- Quarto CLI ---
            RUN curl -sL https://github.com/quarto-dev/quarto-cli/releases/latest/download/quarto-linux-amd64.deb \
                -o /tmp/quarto.deb \\
                && dpkg -i /tmp/quarto.deb \\
                && rm /tmp/quarto.deb
        """)

    dockerfile = textwrap.dedent(f"""\
        # =============================================================================
        # MORIE Dockerfile -- auto-generated
        # Methods for Observational Inference and Robust Analysis of Interventions in Scientific Experimentation
        # =============================================================================
        # Stage 1: Base Python environment
        FROM python:{python_version}-slim AS base

        ENV DEBIAN_FRONTEND=noninteractive
        ENV PYTHONUNBUFFERED=1
        ENV MORIE_CONTAINER=1

        WORKDIR {DEFAULT_WORKDIR}

        # System dependencies
        RUN apt-get update && apt-get install -y --no-install-recommends \\
            build-essential curl git \\
            && rm -rf /var/lib/apt/lists/*

        {r_block}
        {quarto_block}
        # --- Python package ---
        COPY pyproject.toml README.md ./
        COPY py-package/ py-package/
        RUN pip install --no-cache-dir -e ".{extras_str}"

        # Copy remaining project files
        COPY . .

        # Data and output directories
        RUN mkdir -p {DEFAULT_DATA_MOUNT} {DEFAULT_OUTPUT_MOUNT}

        EXPOSE 8888
        CMD ["python", "-m", "morie.runner", "--help"]
    """)

    if output_path is not None:
        Path(output_path).write_text(dockerfile)
        logger.info("Wrote Dockerfile to %s", output_path)

    return dockerfile


def generate_compose_file(
    *,
    include_ollama: bool = True,
    output_path: str | Path | None = None,
) -> str:
    """Generate a docker-compose.yml for MORIE services.

    Parameters
    ----------
    include_ollama : bool
        Include Ollama LLM sidecar service.
    output_path : str | Path | None
        If provided, write the file to this path.

    Returns
    -------
    str
        docker-compose.yml contents.
    """
    ollama_service = ""
    if include_ollama:
        ollama_service = textwrap.dedent("""\
          ollama:
            image: ollama/ollama:latest
            ports:
              - "11434:11434"
            volumes:
              - ollama-models:/root/.ollama
            networks:
              - morie-net
        """)

    compose = textwrap.dedent(f"""\
        # MORIE Docker Compose -- auto-generated
        version: "3.8"

        services:
          morie:
            build: .
            image: {MORIE_IMAGE_NAME}:{MORIE_IMAGE_TAG}
            volumes:
              - ./data:/morie/data:ro
              - ./output:/morie/output
            environment:
              - MORIE_CONTAINER=1
              - MORIE_OLLAMA_HOST=ollama
            networks:
              - morie-net
            depends_on:
              - ollama

        {ollama_service}
        volumes:
          ollama-models:

        networks:
          morie-net:
            driver: bridge
    """)

    if output_path is not None:
        Path(output_path).write_text(compose)
        logger.info("Wrote docker-compose.yml to %s", output_path)

    return compose


# ---------------------------------------------------------------------------
# Rich rendering helpers
# ---------------------------------------------------------------------------


def render_health_check(result: HealthCheckResult) -> None:
    """Display health check results with rich formatting.

    Parameters
    ----------
    result : HealthCheckResult
        Health check to render.
    """
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title="MORIE Container Health Check",
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        table.add_column("Check", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Detail")

        for check in result.checks:
            status = "[green]PASS[/green]" if check["ok"] else "[red]FAIL[/red]"
            table.add_row(check["name"], status, str(check["detail"]))

        console.print(table)

        if result.all_ok:
            console.print("\n[bold green]All health checks passed.[/bold green]")
        else:
            console.print("\n[bold red]Some health checks failed.[/bold red]")

    except ImportError:
        for check in result.checks:
            status = "PASS" if check["ok"] else "FAIL"
            print(f"  [{status}] {check['name']}: {check['detail']}")


def render_containers(containers: list[ContainerInfo]) -> None:
    """Display container list with rich formatting.

    Parameters
    ----------
    containers : list[ContainerInfo]
        Containers to display.
    """
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title="MORIE Containers",
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        table.add_column("ID", style="dim")
        table.add_column("Name", style="bold")
        table.add_column("Image")
        table.add_column("Status")
        table.add_column("Created")

        for c in containers:
            status_style = "green" if "Up" in c.status else "red"
            table.add_row(
                c.container_id[:12],
                c.name,
                c.image,
                f"[{status_style}]{c.status}[/{status_style}]",
                c.created,
            )

        console.print(table)

    except ImportError:
        for c in containers:
            print(f"  {c.container_id[:12]}  {c.name:<30}  {c.status}")


def render_images(images: list[ImageInfo]) -> None:
    """Display image list with rich formatting.

    Parameters
    ----------
    images : list[ImageInfo]
        Images to display.
    """
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title="MORIE Images",
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        table.add_column("ID", style="dim")
        table.add_column("Repository", style="bold")
        table.add_column("Tag")
        table.add_column("Size", justify="right")
        table.add_column("Created")

        for img in images:
            table.add_row(img.image_id[:12], img.repository, img.tag, img.size, img.created)

        console.print(table)

    except ImportError:
        for img in images:
            print(f"  {img.image_id[:12]}  {img.repository}:{img.tag}  {img.size}")


def render_environment_diff(diff: EnvironmentDiff) -> None:
    """Display environment diff with rich formatting.

    Parameters
    ----------
    diff : EnvironmentDiff
        Environment comparison to render.
    """
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.print("\n[bold cyan]Environment Comparison[/bold cyan]\n")

        overview = Table(box=box.SIMPLE, show_header=False)
        overview.add_column("Property", style="bold")
        overview.add_column("Host")
        overview.add_column("Container")

        py_match = diff.host_python == diff.container_python
        overview.add_row(
            "Python",
            diff.host_python,
            f"[{'green' if py_match else 'yellow'}]{diff.container_python}[/]",
        )
        r_match = diff.host_r == diff.container_r
        overview.add_row(
            "R",
            diff.host_r,
            f"[{'green' if r_match else 'yellow'}]{diff.container_r}[/]",
        )
        overview.add_row("OS", diff.host_os, diff.container_os)
        console.print(overview)

        if diff.package_diffs:
            console.print(f"\n[bold]Version differences ({len(diff.package_diffs)} packages):[/bold]")
            pkg_table = Table(box=box.SIMPLE_HEAVY, header_style="bold")
            pkg_table.add_column("Package")
            pkg_table.add_column("Host")
            pkg_table.add_column("Container")
            for d in diff.package_diffs[:30]:
                pkg_table.add_row(d["package"], d["host"], d["container"])
            console.print(pkg_table)

        if diff.missing_in_container:
            console.print(
                f"\n[yellow]Missing in container ({len(diff.missing_in_container)}):[/yellow] "
                + ", ".join(diff.missing_in_container[:20])
            )
        if diff.missing_on_host:
            console.print(
                f"\n[yellow]Missing on host ({len(diff.missing_on_host)}):[/yellow] "
                + ", ".join(diff.missing_on_host[:20])
            )

    except ImportError:
        print(f"Python: host={diff.host_python}, container={diff.container_python}")
        print(f"R: host={diff.host_r}, container={diff.container_r}")
        print(f"OS: host={diff.host_os}, container={diff.container_os}")
        print(f"Package diffs: {len(diff.package_diffs)}")
