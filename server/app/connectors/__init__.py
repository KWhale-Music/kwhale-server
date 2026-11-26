"""Connector registry.

The public repository ships only the connector contract. Real service
connectors are loaded lazily from the gitignored ``app.connectors_local``
overlay, so they never enter the public tree.
"""

from __future__ import annotations

from .base import Connector, StreamInfo, TrackMeta

REGISTRY: dict[str, Connector] = {}
_OVERLAY_LOADED = False


def register_connector(connector: Connector) -> None:
    """Register a connector under its canonical name."""
    REGISTRY[connector.name] = connector


def _load_overlay() -> None:
    global _OVERLAY_LOADED
    if _OVERLAY_LOADED:
        return
    _OVERLAY_LOADED = True
    try:
        from app import connectors_local
    except ModuleNotFoundError as exc:
        if exc.name == "app.connectors_local":
            return
        raise
    connectors_local.register_all()


def get_connectors() -> dict[str, Connector]:
    _load_overlay()
    return REGISTRY


def get_connector(name: str) -> Connector | None:
    return get_connectors().get(name)


__all__ = [
    "REGISTRY",
    "Connector",
    "StreamInfo",
    "TrackMeta",
    "register_connector",
    "get_connector",
    "get_connectors",
]
