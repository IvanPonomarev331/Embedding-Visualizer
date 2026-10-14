"""Read bounded slices of static YT tables into portable local datasets.

The optional YT SDK is imported only when a remote import is requested without
an injected client. No token is accepted, printed, or persisted by this adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from importlib import import_module
from itertools import islice
import json
from numbers import Integral
from pathlib import Path
from typing import Any

from embedding_visualizer.data import DatasetManifest, DatasetValidationError, write_dataset

from .npy import _normalize_ids


_COLUMN_NAMES = {"id", "name", "embedding", "metadata", "embedding_size"}


def _new_client(proxy: str) -> Any:
    try:
        yt = import_module("yt.wrapper")
    except ImportError:
        raise DatasetValidationError(
            "YT import needs the optional YT Python SDK. Install ytsaurus-client "
            "in this Python environment (python -m pip install ytsaurus-client), "
            "or use your existing compatible yt.wrapper installation. "
            "Local Parquet/NumPy workflows do not need this dependency."
        ) from None
    try:
        # Explicit YtClient instances do not inherit YT_* environment settings.
        # The SDK global configuration has already loaded its normal environment
        # and config file settings; copying it preserves standard authentication.
        return yt.YtClient(proxy=proxy, config=deepcopy(yt.config.get_config(None)))
    except Exception:
        raise DatasetValidationError(
            "Cannot initialize the YT client. Check the installed SDK and its "
            "normal environment/configuration. Client details were not printed."
        ) from None


def _check_request(
    proxy: str, table_path: str, columns: Mapping[str, str], limit: int, offset: int
) -> dict[str, str]:
    if not isinstance(proxy, str) or not proxy.strip():
        raise DatasetValidationError("YT proxy must be a non-empty cluster name or address.")
    if not isinstance(table_path, str) or not table_path.startswith("//") or table_path == "//":
        raise DatasetValidationError("YT table_path must be an absolute //path/to/table.")
    # Supply bounds/column selection only through the explicit parameters. Rich
    # YPath selectors could otherwise replace or combine with the requested slice.
    if any(character in table_path for character in "{}[]<>\n\r") or "/@" in table_path:
        raise DatasetValidationError(
            "Use a plain YT table path without attributes, ranges, or column selectors."
        )
    if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1:
        raise DatasetValidationError("YT limit must be a positive integer.")
    if isinstance(offset, bool) or not isinstance(offset, Integral) or offset < 0:
        raise DatasetValidationError("YT offset must be a non-negative integer.")
    if not isinstance(columns, Mapping):
        raise DatasetValidationError("YT columns must map canonical names to source column names.")
    columns = dict(columns)
    unknown = set(columns) - _COLUMN_NAMES
    missing = {"id", "embedding"} - set(columns)
    if unknown or missing:
        raise DatasetValidationError(
            "YT columns require id and embedding; allowed optional keys are "
            "name, metadata, and embedding_size."
        )
    if any(not isinstance(value, str) or not value.strip() for value in columns.values()):
        raise DatasetValidationError("Every YT source column name must be a non-empty string.")
    return columns


def _read_slice(
    client: Any, table_path: str, columns: list[str], limit: int, offset: int
) -> list[Mapping[str, Any]]:
    stream = None
    try:
        if client.get(f"{table_path}/@type") != "table":
            raise DatasetValidationError("The YT source must be a table.")
        if client.get(f"{table_path}/@dynamic"):
            raise DatasetValidationError(
                "YT import currently supports static tables only. Export a static "
                "snapshot of a dynamic table before importing it."
            )
        path = client.TablePath(
            table_path,
            columns=columns,
            start_index=int(offset),
            end_index=int(offset) + int(limit),
        )
        # JSON avoids requiring optional native YSON bindings on local machines.
        stream = client.read_table(path, format="json", raw=False, unordered=False)
        # The remote range bounds the request; islice also protects local memory
        # if an injected client yields more rows than the requested range.
        return list(islice(stream, int(limit)))
    except DatasetValidationError:
        raise
    except Exception:
        # SDK exceptions can contain HTTP headers or authentication config. Keep
        # them out of both the displayed message and chained traceback.
        raise DatasetValidationError(
            "YT table read failed. Check the proxy, table path, mapped column names, "
            "read permissions, and your SDK authentication/environment. "
            "Remote exception details were not printed."
        ) from None
    finally:
        if stream is not None:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    # A secondary stream cleanup error must not expose SDK
                    # credentials or hide the useful validation/read result.
                    pass


def _metadata(value: Any, row_number: int) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            raise DatasetValidationError(
                f"YT row {row_number}: metadata must contain a JSON object."
            ) from None
    if not isinstance(value, Mapping):
        raise DatasetValidationError(f"YT row {row_number}: metadata must be an object or null.")
    return dict(value)


def import_yt_dataset(
    proxy: str,
    table_path: str,
    output: str | Path,
    *,
    columns: Mapping[str, str],
    dataset_name: str,
    space_id: str,
    version: str,
    default_metric: str = "cosine",
    limit: int = 10_000,
    offset: int = 0,
    overwrite: bool = False,
    client: Any = None,
) -> Path:
    """Import the deterministic row interval ``[offset, offset + limit)``.

    Only explicitly mapped columns are requested. Integer IDs are converted to
    strings, matching the NumPy adapter; existing strings keep their leading
    zeroes. ``embedding_size``, when mapped, must agree with each vector and the
    inferred dataset dimension. Empty slices fail without creating output.

    ``client`` is an optional configured YtClient-compatible instance, useful for
    existing SDK integrations and offline testing. Its configuration takes
    precedence over ``proxy``; callers must supply a client for that cluster.
    """
    columns = _check_request(proxy, table_path, columns, limit, offset)
    # Validate manifest identity/metric before any remote access; dimension will
    # be inferred from the first vector and then checked for every row.
    DatasetManifest(dataset_name, space_id, version, 1, default_metric)
    output = Path(output).expanduser()
    if output.exists() and not overwrite:
        raise DatasetValidationError(
            f"Output already exists: {output}. Choose a new path or enable overwrite."
        )
    active_client = client if client is not None else _new_client(proxy)
    rows = _read_slice(
        active_client, table_path, list(dict.fromkeys(columns.values())), limit, offset
    )
    if not rows:
        raise DatasetValidationError(
            "The requested YT row slice is empty. Check the table and offset."
        )
    canonical: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise DatasetValidationError(f"YT row {offset + index}: expected an object.")
        missing = [name for name in columns.values() if name not in row]
        if missing:
            raise DatasetValidationError(
                f"YT row {offset + index}: mapped source columns are missing: "
                + ", ".join(missing)
            )
        canonical.append({name: row[source] for name, source in columns.items()})

    first_vector = canonical[0]["embedding"]
    if not isinstance(first_vector, (list, tuple)) or not first_vector:
        raise DatasetValidationError("YT embedding must be a non-empty numeric vector in every row.")
    dimension = len(first_vector)
    manifest = DatasetManifest(dataset_name, space_id, version, dimension, default_metric)
    ids = _normalize_ids([row["id"] for row in canonical], "YT")
    if "embedding_size" in columns:
        for index, row in enumerate(canonical):
            size = row["embedding_size"]
            if isinstance(size, bool) or not isinstance(size, Integral) or size != dimension:
                raise DatasetValidationError(
                    f"YT row {offset + index}: embedding_size must be integer {dimension}."
                )
    return write_dataset(
        output,
        ids,
        [row["embedding"] for row in canonical],
        manifest,
        names=[row.get("name") for row in canonical],
        metadata=[_metadata(row.get("metadata"), offset + index) for index, row in enumerate(canonical)],
        overwrite=overwrite,
    )
