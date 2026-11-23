from __future__ import annotations

import os as _os

from .py.core.startup import install_missing_requirements as _install_missing_requirements

_install_missing_requirements(_os.path.dirname(_os.path.realpath(__file__)))

del _install_missing_requirements

from .py.server.ne_server import *
from .py.nodes.registration import register_nodes as _register_nodes
from .registry import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

_register_nodes()

del _register_nodes

# Diagnostics, off unless asked for: eng/profile_nodes.py prints wall and CPU time per node, and per stage
# inside the detailer. Guarded against ModuleNotFoundError because eng/ is not carried by the public mirror.
if _os.environ.get("NE_TOOLS_PROFILE", "").strip().lower() in ("1", "true", "yes", "on"):
    try:
        from .eng.profile_nodes import install as _install_profiling
    except ModuleNotFoundError:
        _install_profiling = None

    if _install_profiling is not None:
        from .py.nodes import detailer as _detailer

        _install_profiling(_detailer)
        del _detailer

    del _install_profiling

del _os

WEB_DIRECTORY = "js"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
