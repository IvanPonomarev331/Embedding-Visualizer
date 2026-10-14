"""Safe local caches, portable study archives, and selected-row exports.

The archive format contains NumPy arrays plus UTF-8 JSON, never pickle. Cache
entries are immutable: concurrent writers publish one complete file and retain
whichever valid entry won the race. Study saves atomically replace their target.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import numbers
import os
from pathlib import Path
import re
import tempfile
import zipfile
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


_ARCHIVE_VERSION = 1
_CACHE_VERSION = 1
_MAX_ARCHIVE_BYTES = 512 * 1024**2
_HEADER_NAME = "__header__"


class PersistenceError(ValueError):
    """Invalid or incompatible persisted data, with an actionable message."""


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise PersistenceError("JSON state must not contain NaN or infinity.")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise PersistenceError("JSON object keys must be strings.")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise PersistenceError(f"State contains a value that is not JSON-compatible: {type(value).__name__}.")


def _json_bytes(value) -> bytes:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _object(value, name):
    if not isinstance(value, Mapping):
        raise PersistenceError(f"{name} must be a JSON object.")
    return _json_value(value)


def _indices(indices, row_count: int | None = None) -> np.ndarray:
    array = np.asarray(indices)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise PersistenceError("Row indices must be a one-dimensional integer sequence.")
    if not isinstance(indices, np.ndarray):
        original = np.asarray(indices, dtype=object)
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
               for value in original.flat):
            raise PersistenceError("Row indices must be a one-dimensional integer sequence, not booleans.")
    if np.any(array < 0) or (row_count is not None and np.any(array >= row_count)):
        raise PersistenceError("Row indices are outside the dataset.")
    if array.size and np.max(array) > np.iinfo(np.int64).max:
        raise PersistenceError("Row indices exceed the supported integer range.")
    return array.astype(np.int64, copy=False)


def _hash_part(digest, data):
    digest.update(len(data).to_bytes(8, "little"))
    digest.update(data)


def dataset_fingerprint(dataset) -> str:
    """Hash dataset contents and identity, including row ordering and metadata."""
    digest = hashlib.sha256(b"embedding-visualizer-dataset-v1")
    _hash_part(digest, _json_bytes(dataset.manifest.to_dict()))
    for value in (dataset.ids, dataset.names, dataset.metadata):
        _hash_part(digest, _json_bytes(value))
    matrix = np.ascontiguousarray(dataset.embeddings)
    if matrix.dtype.hasobject:
        raise PersistenceError("Dataset embeddings cannot use object dtype.")
    _hash_part(digest, _json_bytes({"dtype": matrix.dtype.str, "shape": matrix.shape}))
    _hash_part(digest, memoryview(matrix).cast("B"))
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _dependency_versions():
    versions = {}
    for name in ("numpy", "scikit-learn", "umap-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _safe_array(value, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"Array {name!r} must be a rectangular NumPy-compatible array.") from exc
    if array.dtype.kind not in "biufSU" or array.dtype.hasobject:
        raise PersistenceError(f"Array {name!r} must contain real numbers, booleans or fixed-width strings; object arrays are forbidden.")
    if array.dtype.kind == "f" and np.isinf(array).any():
        raise PersistenceError(f"Array {name!r} contains infinity.")
    return array


def _archive_bytes(kind: str, state: Mapping, arrays: Mapping) -> bytes:
    state = _object(state, "State")
    if not isinstance(arrays, Mapping) or not all(isinstance(name, str) and name for name in arrays):
        raise PersistenceError("Arrays must be a mapping with nonempty string names.")
    names = sorted(arrays)
    values = [_safe_array(arrays[name], name) for name in names]
    header = _json_bytes({"format": "embedding-visualizer", "version": _ARCHIVE_VERSION,
                          "kind": kind, "state": state, "arrays": names})
    if len(header) + sum(value.nbytes for value in values) > _MAX_ARCHIVE_BYTES:
        raise PersistenceError("Archive data exceed the 512 MiB limit; save fewer objects or arrays.")
    payload = {f"array_{index}": value for index, value in enumerate(values)}
    payload[_HEADER_NAME] = np.frombuffer(header, dtype=np.uint8)
    output = io.BytesIO()
    np.savez_compressed(output, **payload)
    return output.getvalue()


def _read_archive(data: bytes, kind: str):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise PersistenceError("Archive content must be bytes.")
    if len(data) > _MAX_ARCHIVE_BYTES:
        raise PersistenceError("Archive exceeds the 512 MiB limit.")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            if len(members) > 1000 or len({member.filename for member in members}) != len(members):
                raise PersistenceError("Archive contains duplicate or too many entries.")
            if sum(member.file_size for member in members) > _MAX_ARCHIVE_BYTES:
                raise PersistenceError("Expanded archive exceeds the 512 MiB limit.")
            if any(not re.fullmatch(r"(?:__header__|array_\d+)\.npy", member.filename) for member in members):
                raise PersistenceError("Archive contains unexpected files.")
            # Check shapes before np.load allocates arrays. A tiny corrupt NPY
            # member can otherwise claim billions of elements in its header.
            for member in members:
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise PersistenceError("Unsupported NumPy array format in archive.")
                    if dtype.hasobject or dtype.kind not in "biufSU":
                        raise PersistenceError("Unsafe array dtype in archive; object arrays are forbidden.")
                    if not all(isinstance(size, int) and size >= 0 for size in shape):
                        raise PersistenceError("Invalid array shape in archive.")
                    array_bytes = math.prod(shape) * dtype.itemsize
                    if array_bytes > _MAX_ARCHIVE_BYTES or stream.tell() + array_bytes != member.file_size:
                        raise PersistenceError("Array shape does not match its stored size or exceeds the archive limit.")
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            raw_header = archive[_HEADER_NAME]
            if raw_header.dtype != np.uint8 or raw_header.ndim != 1:
                raise PersistenceError("Archive JSON header has an invalid representation.")
            header = json.loads(raw_header.tobytes().decode("utf-8"))
            if not isinstance(header, dict) or header.get("format") != "embedding-visualizer":
                raise PersistenceError("This is not an Embedding Visualizer archive.")
            if type(header.get("version")) is not int or header["version"] != _ARCHIVE_VERSION:
                raise PersistenceError("Unsupported archive version.")
            if header.get("kind") != kind:
                raise PersistenceError(f"Expected a {kind} archive.")
            state = _object(header.get("state"), "Archive state")
            names = header.get("arrays")
            if not isinstance(names, list) or not all(isinstance(name, str) and name for name in names) or len(set(names)) != len(names):
                raise PersistenceError("Archive array names are invalid.")
            expected = {_HEADER_NAME} | {f"array_{index}" for index in range(len(names))}
            if set(archive.files) != expected:
                raise PersistenceError("Archive arrays do not match its header.")
            arrays = {name: _safe_array(archive[f"array_{index}"], name).copy()
                      for index, name in enumerate(names)}
        return {"state": state, "arrays": arrays}
    except PersistenceError:
        raise
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, zipfile.BadZipFile, zlib.error, EOFError, OverflowError) as exc:
        raise PersistenceError(f"Cannot read archive: {exc}") from exc


def _read_file(path) -> bytes:
    path = Path(path).expanduser()
    if path.stat().st_size > _MAX_ARCHIVE_BYTES:
        raise PersistenceError("Archive exceeds the 512 MiB limit.")
    return path.read_bytes()


def _atomic_write(path: Path, data: bytes, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                # Atomic create-if-absent: a concurrent writer cannot expose a
                # partially written file or replace an existing immutable key.
                os.link(temporary, path)
            except FileExistsError:
                pass
    finally:
        temporary.unlink(missing_ok=True)


class ComputeCache:
    """Versioned content-addressed cache of safe arrays plus JSON metadata."""

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir).expanduser()

    def key(self, fingerprint: str, operation: str, params: Mapping, indices=None) -> str:
        if not isinstance(fingerprint, str) or not fingerprint:
            raise PersistenceError("A nonempty dataset fingerprint is required.")
        if not isinstance(operation, str) or not operation:
            raise PersistenceError("A nonempty operation name is required.")
        value = {"version": _CACHE_VERSION, "dependencies": _dependency_versions(),
                 "fingerprint": fingerprint, "operation": operation, "params": _object(params, "Parameters"),
                 "indices": None if indices is None else _indices(indices).tolist()}
        return hashlib.sha256(_json_bytes(value)).hexdigest()

    def _path(self, key: str) -> Path:
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
            raise PersistenceError("Cache key must be a 64-character lowercase SHA-256 hex string.")
        return self.cache_dir / key[:2] / f"{key}.npz"

    def load(self, key: str) -> dict | None:
        path = self._path(key)
        try:
            result = _read_archive(_read_file(path), "cache")
            envelope = result["state"]
            if envelope.get("key") != key or not isinstance(envelope.get("metadata"), dict):
                return None
            return {"arrays": result["arrays"], "metadata": envelope["metadata"]}
        except (OSError, PersistenceError):
            return None

    def save(self, key: str, arrays: Mapping, metadata: Mapping) -> None:
        path = self._path(key)
        metadata = _object(metadata, "Metadata")
        data = _archive_bytes("cache", {"key": key, "metadata": metadata}, arrays)
        _atomic_write(path, data, replace=False)


def study_bytes(state: Mapping, arrays: Mapping) -> bytes:
    """Serialize JSON study settings with actual coordinates/labels if supplied."""
    return _archive_bytes("study", state, arrays)


def read_study_bytes(data: bytes) -> dict:
    return _read_archive(data, "study")


def save_study(path, state: Mapping, arrays: Mapping) -> Path:
    path = Path(path).expanduser()
    _atomic_write(path, study_bytes(state, arrays), replace=True)
    return path


def load_study(path) -> dict:
    try:
        return read_study_bytes(_read_file(path))
    except OSError as exc:
        raise PersistenceError(f"Cannot read study: {exc}") from exc


def export_rows(dataset, indices, coordinates=None, labels=None, format: str = "csv") -> bytes:
    """Export ordered rows with explicit IDs and JSON metadata.

    Optional coordinate and label arrays are aligned with the complete dataset,
    not the exported slice. Embeddings are deliberately omitted from this
    inspection export; the source dataset remains the embedding interchange.
    """
    if format not in ("csv", "parquet"):
        raise PersistenceError("Export format must be 'csv' or 'parquet'.")
    row_count = len(dataset.ids)
    rows = _indices(indices, row_count)
    fields = [pa.field("id", pa.string()), pa.field("name", pa.string()), pa.field("metadata", pa.string())]
    columns = {
        "id": [dataset.ids[index] for index in rows],
        "name": [dataset.names[index] for index in rows],
        "metadata": [_json_bytes(dataset.metadata[index]).decode("utf-8") for index in rows],
    }
    if coordinates is not None:
        coordinates = _safe_array(coordinates, "coordinates")
        if coordinates.ndim != 2 or coordinates.shape[0] != row_count or coordinates.dtype.kind not in "iuf":
            raise PersistenceError("Coordinates must be a numeric 2D array aligned with all dataset rows.")
        for dimension in range(coordinates.shape[1]):
            name = f"coord{dimension}"
            fields.append(pa.field(name, pa.float64()))
            columns[name] = [None if np.isnan(value) else float(value)
                             for value in coordinates[rows, dimension]]
    if labels is not None:
        labels = _safe_array(labels, "labels")
        if labels.shape != (row_count,) or labels.dtype.kind not in "iu":
            raise PersistenceError("Labels must be a one-dimensional integer array aligned with all dataset rows.")
        fields.append(pa.field("cluster", pa.int64()))
        columns["cluster"] = [None if value == -2 else int(value) for value in labels[rows]]
    if format == "parquet":
        table = pa.Table.from_pydict(columns, schema=pa.schema(fields))
        output = io.BytesIO()
        pq.write_table(table, output)
        return output.getvalue()
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(zip(*columns.values()))
    return output.getvalue().encode("utf-8")
