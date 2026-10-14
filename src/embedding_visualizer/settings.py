"""Portable configuration; loading settings never creates directories."""

from dataclasses import dataclass, replace
import os
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    cache_dir: Path
    address: str = "127.0.0.1"
    port: int = 8501


def load_settings(config: Path | None = None) -> Settings:
    values = {}
    base = Path.cwd()
    if config is not None:
        config = config.expanduser().resolve()
        with config.open("rb") as handle:
            values = tomllib.load(handle)
        base = config.parent
    allowed = {"data_dir", "cache_dir", "address", "port"}
    if unknown := values.keys() - allowed:
        raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
    for key in allowed:
        if (value := os.environ.get(f"VISUALIZER_{key.upper()}")) is not None:
            values[key] = int(value) if key == "port" else value
    paths = {}
    for key, default in (("data_dir", "data"), ("cache_dir", ".cache")):
        raw = values.get(key, default)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{key} must be a non-empty path string")
        path = Path(raw).expanduser()
        paths[key] = (path if path.is_absolute() else base / path).resolve()
    port = values.get("port", 8501)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("port must be an integer from 1 to 65535")
    address = values.get("address", "127.0.0.1")
    if not isinstance(address, str) or not address.strip():
        raise ValueError("address must be a non-empty string")
    return Settings(**paths, address=address, port=port)


def override_server(settings: Settings, address: str | None, port: int | None) -> Settings:
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("port must be from 1 to 65535")
    if address is not None and not address.strip():
        raise ValueError("address must be non-empty")
    return replace(settings, address=address or settings.address, port=port or settings.port)
