from __future__ import annotations

from dataclasses import dataclass

from .constants import NODE_PREFIX

NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}


@dataclass(frozen=True, slots=True)
class NodeRegistration:
    name: str
    node_class: type
    display_name: str | None = None


def normalize_node_name(name: str) -> str:
    if not name:
        raise ValueError("Node name must not be empty.")

    return name if name.startswith(NODE_PREFIX) else f"{NODE_PREFIX}{name}"


def register_node(registration: NodeRegistration) -> None:
    node_id = normalize_node_name(registration.name)

    if node_id in NODE_CLASS_MAPPINGS:
        raise ValueError(f"Node is already registered: {node_id}")

    NODE_CLASS_MAPPINGS[node_id] = registration.node_class

    if registration.display_name is not None:
        NODE_DISPLAY_NAME_MAPPINGS[node_id] = registration.display_name


def register_nodes(registrations: list[NodeRegistration]) -> None:
    NODE_CLASS_MAPPINGS.clear()
    NODE_DISPLAY_NAME_MAPPINGS.clear()

    for registration in registrations:
        register_node(registration)
