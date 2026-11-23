from __future__ import annotations

import importlib.metadata
import os
import re
import subprocess
import sys
from pathlib import Path

_DISABLE_MODES = {"0", "false", "no", "off"}
_INSTALL_MODES = {"1", "true", "yes", "on", "install"}
# Every variant installs the same `onnxruntime` module, so having any one of them satisfies the requirement
# and installing a second one on top would shadow the first.
_ONNXRUNTIME_DISTRIBUTIONS = (
    "onnxruntime-gpu",
    "onnxruntime",
    "onnxruntime-directml",
    "onnxruntime-rocm",
    "onnxruntime-silicon",
)
_PACKAGE_ALTERNATIVES: dict[str, tuple[str, ...]] = {
    "onnxruntime-gpu": _ONNXRUNTIME_DISTRIBUTIONS,
    "onnxruntime": _ONNXRUNTIME_DISTRIBUTIONS,
}


def install_missing_requirements(package_root: str | os.PathLike[str]) -> None:
    """Check runtime requirements and warn (default) or install when missing.

    Controlled by NE_TOOLS_AUTO_INSTALL:
    - warn (default): prints missing packages without modifying the environment.
    - 1 / true / yes / on: installs requirements.txt via pip when packages are missing.
    - 0 / false / no / off: disables the startup check altogether.
    """
    mode = os.environ.get("NE_TOOLS_AUTO_INSTALL", "warn").strip().lower()
    if mode in _DISABLE_MODES:
        return

    requirements_path = Path(package_root) / "requirements.txt"
    if not requirements_path.is_file():
        return

    missing = _find_missing_requirements(requirements_path)
    if not missing:
        return

    if mode not in _INSTALL_MODES:
        print(
            "[comfyui-ne-tools] Missing Python requirements: "
            + ", ".join(missing)
            + ". Install requirements.txt in the current ComfyUI venv (or set NE_TOOLS_AUTO_INSTALL=1)."
        )
        return

    print(
        "[comfyui-ne-tools] Installing missing Python requirements into current environment: "
        + ", ".join(missing)
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])


def _find_missing_requirements(requirements_path: Path) -> list[str]:
    missing: list[str] = []

    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        package_name = _parse_requirement_line(raw_line)
        if not package_name:
            continue

        if not _is_package_installed(package_name):
            missing.append(package_name)

    return missing


def _parse_requirement_line(line: str) -> str | None:
    stripped = line.split("#", 1)[0].strip()
    if not stripped or stripped.startswith("-"):
        return None

    spec, separator, marker = stripped.partition(";")
    if separator and not _applies_to_current_environment(marker):
        return None

    match = re.match(r"^([A-Za-z0-9_.-]+)", spec.strip())
    return match.group(1) if match else None


def _applies_to_current_environment(marker: str) -> bool:
    try:
        from packaging.markers import Marker

        return Marker(marker).evaluate()
    except ImportError:
        match = re.fullmatch(r'sys_platform\s*(==|!=)\s*["\']([^"\']+)["\']', marker.strip())
        if match:
            operator, platform_name = match.groups()
            return (sys.platform == platform_name) if operator == "==" else (sys.platform != platform_name)

        return True


def _is_package_installed(package_name: str) -> bool:
    candidates = _PACKAGE_ALTERNATIVES.get(package_name.lower(), (package_name,))
    return any(_has_distribution(candidate) for candidate in candidates)


def _has_distribution(package_name: str) -> bool:
    try:
        importlib.metadata.distribution(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False
