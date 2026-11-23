from __future__ import annotations

import os
import sys

import folder_paths
from aiohttp import web
from server import PromptServer

from ...constants import (
    AUTOCOMPLETE_RELATIVE_PATH,
    FOLDER_LORAS,
    QUERY_EXPERTS,
    REBOOT_HEADER,
    REBOOT_HEADER_VALUE,
    ROUTE_AUTOCOMPLETE,
    ROUTE_LORAS,
    ROUTE_LORA_INFO,
    ROUTE_LORA_PREVIEW,
    ROUTE_REBOOT,
    ROUTE_RESET,
)
from ..core.base import get_path_by_filename
from ..core.models import expert_names, model_names
from .state import reset_registry


def _extension_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


LORA_PREVIEW_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def _lora_path(lora: str) -> str | None:
    lora_name = get_path_by_filename(lora, FOLDER_LORAS)
    if lora_name is None:
        return None

    lora_path = folder_paths.get_full_path(FOLDER_LORAS, lora_name)
    if not lora_path:
        return None

    return os.path.realpath(lora_path)


def _lora_sidecar_path(lora: str, extension: str) -> str | None:
    lora_path = _lora_path(lora)
    if lora_path is None:
        return None

    sidecar_path = os.path.splitext(lora_path)[0] + extension
    lora_dir = os.path.dirname(lora_path)
    if os.path.dirname(os.path.realpath(sidecar_path)) != lora_dir:
        return None

    return sidecar_path


def _lora_sidecar_info_path(lora: str) -> str | None:
    return _lora_sidecar_path(lora, ".txt")


def _lora_sidecar_preview_path(lora: str) -> str | None:
    for extension in LORA_PREVIEW_EXTENSIONS:
        preview_path = _lora_sidecar_path(lora, extension)
        if preview_path and os.path.isfile(preview_path):
            return preview_path

    return None


def _read_text_file(path: str) -> str:
    encodings = ("utf-8", "utf-8-sig", "cp1251")
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError as error:
            last_error = error

    if last_error:
        raise last_error

    with open(path, "r") as file:
        return file.read()

def _autocomplete_file_path() -> str:
    return os.path.join(_extension_root(), *AUTOCOMPLETE_RELATIVE_PATH)


@PromptServer.instance.routes.get(ROUTE_AUTOCOMPLETE)
async def get_autocomplete(request):
    autocomplete_file = _autocomplete_file_path()

    if os.path.isfile(autocomplete_file):
        return web.FileResponse(autocomplete_file)

    return web.Response(status=404)


@PromptServer.instance.routes.get(ROUTE_LORAS)
async def get_loras(request):
    if request.query.get(QUERY_EXPERTS):
        return web.json_response(expert_names(FOLDER_LORAS))

    return web.json_response(model_names(FOLDER_LORAS))


@PromptServer.instance.routes.get(ROUTE_LORA_INFO)
async def get_lora_info(request):
    lora = request.query.get("lora")
    if not lora:
        return web.json_response({"has_info": False, "has_preview": False})

    info_path = _lora_sidecar_info_path(lora)
    preview_path = _lora_sidecar_preview_path(lora)

    has_info = bool(info_path and os.path.isfile(info_path))
    has_preview = bool(preview_path)

    if not has_info and not has_preview:
        return web.json_response({"has_info": False, "has_preview": False})

    try:
        return web.json_response({
            "has_info": has_info,
            "has_preview": has_preview,
            "filename": os.path.basename(info_path) if has_info else os.path.basename(preview_path),
            "content": _read_text_file(info_path) if has_info else "",
        })
    except Exception as error:
        return web.json_response({"error": str(error)}, status=500)


@PromptServer.instance.routes.get(ROUTE_LORA_PREVIEW)
async def get_lora_preview(request):
    lora = request.query.get("lora")
    if not lora:
        return web.Response(status=404)

    preview_path = _lora_sidecar_preview_path(lora)
    if not preview_path:
        return web.Response(status=404)

    return web.FileResponse(preview_path)


# POST plus a custom header: GET fires from an <img> tag, and a plain <form method="POST"> from another tab
# still skips CORS preflight unless a non-simple header is required.
@PromptServer.instance.routes.post(ROUTE_REBOOT)
async def restart(request):
    if request.headers.get(REBOOT_HEADER) != REBOOT_HEADER_VALUE:
        return web.Response(status=403)

    try:
        sys.stdout.close_log()
    except Exception:
        pass

    if "__COMFY_CLI_SESSION__" in os.environ:
        with open(os.path.join(os.environ["__COMFY_CLI_SESSION__"] + ".reboot"), "w"):
            pass

        print("\nRestarting...\n\n")
        raise SystemExit(0)

    print("\nRestarting... [Legacy Mode]\n\n")

    sys_argv = sys.argv.copy()

    if "--windows-standalone-build" in sys_argv:
        sys_argv.remove("--windows-standalone-build")

    if sys.platform.startswith("win32"):
        commands = [f'"{sys.executable}"', f'"{sys_argv[0]}"'] + sys_argv[1:]
    elif sys_argv[0].endswith("__main__.py"):
        module_name = os.path.basename(os.path.dirname(sys_argv[0]))
        commands = [sys.executable, "-m", module_name] + sys_argv[1:]
    else:
        commands = [sys.executable] + sys_argv

    print(f"Command: {commands}", flush=True)

    return os.execv(sys.executable, commands)


@PromptServer.instance.routes.post(ROUTE_RESET)
async def reset(request):
    try:
        node_id = request.match_info["node_id"]
        reset_registry[node_id] = True

        return web.json_response({"status": "ok"})
    except Exception as error:
        return web.json_response({"error": str(error)}, status=500)