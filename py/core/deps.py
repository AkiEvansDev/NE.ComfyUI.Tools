from __future__ import annotations

from importlib import import_module
from types import ModuleType


def require_dependency(import_name: str, package_name: str | None = None) -> ModuleType:
    try:
        return import_module(import_name)
    except ImportError as error:
        package = package_name or import_name
        raise ImportError(
            f"Missing dependency `{package}`. Install requirements.txt for comfyui-ne-tools."
        ) from error
