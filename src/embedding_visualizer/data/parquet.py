"""Read and write the portable, local Parquet dataset contract.

Metadata is stored as a JSON string, so nested heterogeneous objects survive
round trips without relying on an inferred Arrow struct schema.
"""

from __future__ import annotations

import json
import math
import numbers
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .models import Dataset, DatasetManifest, DatasetValidationError, ValidationReport


class _Issues:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.total = 0

    def add(self, message: str) -> None:
        self.total += 1
        if len(self.messages) < 20:
            self.messages.append(message)

    def raise_if_any(self) -> None:
        if self.total:
            raise DatasetValidationError(self.messages, self.total)


def _context(row: int, identifier: Any) -> str:
    label = repr(identifier)
    if len(label) > 100:
        label = label[:97] + "..."
    return f"row {row + 1} (id={label})"


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _check_json(value: Any, location: str = "metadata") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a nonfinite number")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} contains a non-string object key")
            _check_json(child, f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _check_json(child, f"{location}[{index}]")
        return
    raise ValueError(f"{location} contains non-JSON value of type {type(value).__name__}")


def _validate(
    ids: Any,
    embeddings: Any,
    manifest: DatasetManifest,
    names: Any = None,
    metadata: Any = None,
    embedding_sizes: Any = None,
) -> Dataset:
    if not isinstance(manifest, DatasetManifest):
        raise DatasetValidationError("manifest must be a DatasetManifest")
    issues = _Issues()
    if isinstance(ids, (str, bytes)):
        raise DatasetValidationError("ids must be a sequence of strings")
    try:
        ids = list(ids)
    except TypeError as error:
        raise DatasetValidationError("ids must be a sequence of strings") from error
    count = len(ids)
    if count == 0:
        issues.add("dataset must contain at least one row")
    columns = {"embedding": embeddings, "name": names, "metadata": metadata}
    if embedding_sizes is not None:
        columns["embedding_size"] = embedding_sizes
    converted: dict[str, list[Any]] = {}
    for label, column in columns.items():
        if column is None and label in ("name", "metadata"):
            converted[label] = [None] * count
            continue
        if isinstance(column, (str, bytes)):
            issues.add(f"{label} must be a sequence with one value per row")
            continue
        try:
            converted[label] = list(column)
        except TypeError:
            issues.add(f"{label} must be a sequence with one value per row")
            continue
        if len(converted[label]) != count:
            issues.add(f"{label} has {len(converted[label])} rows; id has {count}")
    issues.raise_if_any()

    matrix = np.empty((count, manifest.embedding_dim), dtype=np.float32)
    clean_metadata: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    fields: dict[str, set[str]] = {}
    zero_ids: list[str] = []
    for index, identifier in enumerate(ids):
        context = _context(index, identifier)
        id_valid = isinstance(identifier, str) and bool(identifier.strip())
        if not id_valid:
            issues.add(f"{context}: id must be a nonempty string")
        elif identifier in seen:
            issues.add(f"{context}: duplicate id (first present at row {seen[identifier] + 1})")
        else:
            seen[identifier] = index

        name = converted["name"][index]
        if name is not None and not isinstance(name, str):
            issues.add(f"{context}: name must be a string or null")

        vector = converted["embedding"][index]
        vector_valid = True
        if not isinstance(vector, (list, tuple, np.ndarray)):
            issues.add(f"{context}: embedding must be a one-dimensional numeric vector")
            vector_valid = False
        elif isinstance(vector, np.ndarray) and vector.ndim != 1:
            issues.add(f"{context}: embedding must be one-dimensional, got shape {vector.shape}")
            vector_valid = False
        elif len(vector) != manifest.embedding_dim:
            issues.add(
                f"{context}: embedding length {len(vector)} does not match "
                f"manifest.embedding_dim {manifest.embedding_dim}"
            )
            vector_valid = False
        if vector_valid:
            for component, value in enumerate(vector):
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
                    issues.add(f"{context}: embedding[{component}] must be a real number, not {type(value).__name__}")
                    vector_valid = False
                    break
                try:
                    finite = math.isfinite(value)
                except (OverflowError, TypeError):
                    finite = False
                if not finite:
                    issues.add(f"{context}: embedding[{component}] must be finite")
                    vector_valid = False
                    break
        if vector_valid:
            try:
                with np.errstate(over="raise", invalid="raise"):
                    matrix[index] = np.asarray(vector, dtype=np.float32)
                if not np.isfinite(matrix[index]).all():
                    raise FloatingPointError
            except (OverflowError, ValueError, FloatingPointError):
                issues.add(f"{context}: embedding contains values outside finite float32 range")
                vector_valid = False
        if vector_valid and id_valid and not np.any(matrix[index]):
            zero_ids.append(identifier)

        if embedding_sizes is not None:
            size = converted["embedding_size"][index]
            if isinstance(size, bool) or not isinstance(size, numbers.Integral) or size != manifest.embedding_dim:
                issues.add(f"{context}: embedding_size must be integer {manifest.embedding_dim}, got {size!r}")

        value = converted["metadata"][index]
        if value is None:
            value = {}
        try:
            if isinstance(value, str):
                value = json.loads(value)
            if not isinstance(value, dict):
                raise ValueError("metadata must be a JSON object or null")
            _check_json(value)
            # Detach caller-owned nested objects and use ordinary JSON values.
            value = json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))
            clean_metadata.append(value)
            for key, child in value.items():
                fields.setdefault(key, set()).add(_json_type(child))
        except (ValueError, TypeError, RecursionError) as error:
            issues.add(f"{context}: invalid metadata: {error}")
            clean_metadata.append({})

    issues.raise_if_any()
    warnings: list[str] = []
    if zero_ids and manifest.default_metric == "cosine":
        warnings.append(
            f"{len(zero_ids)} zero vector(s): cosine similarity is undefined for these rows. "
            "Exclude them or explicitly choose a supported zero-vector policy before cosine search."
        )
    return Dataset(
        ids=ids,
        names=converted["name"],
        embeddings=matrix,
        metadata=clean_metadata,
        manifest=manifest,
        report=ValidationReport(
            row_count=count,
            embedding_dim=manifest.embedding_dim,
            zero_vector_ids=zero_ids,
            warnings=warnings,
            metadata_fields={key: sorted(types) for key, types in sorted(fields.items())},
        ),
    )


def load_dataset(path: str | Path) -> Dataset:
    """Load a dataset directory or a Parquet file beside ``manifest.json``."""
    path = Path(path).expanduser()
    parquet_path = path / "dataset.parquet" if path.is_dir() else path
    manifest_path = parquet_path.parent / "manifest.json"
    try:
        manifest = DatasetManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetValidationError(f"cannot read manifest {manifest_path}: {error}") from error
    try:
        parquet_file = pq.ParquetFile(parquet_path)
        schema_names = parquet_file.schema_arrow.names
        required = {"id", "embedding"}
        missing = required - set(schema_names)
        if missing:
            raise DatasetValidationError("Parquet missing columns: " + ", ".join(sorted(missing)))
        if len(set(schema_names)) != len(schema_names):
            raise DatasetValidationError("Parquet contains duplicate column names")
        columns = [name for name in ("id", "name", "embedding", "metadata", "embedding_size") if name in schema_names]
        table = parquet_file.read(columns=columns)
        data = {name: table[name].to_pylist() for name in columns}
    except (OSError, pa.ArrowException) as error:
        raise DatasetValidationError(f"cannot read Parquet {parquet_path}: {error}") from error
    dataset = _validate(
        ids=data["id"],
        embeddings=data["embedding"],
        manifest=manifest,
        names=data.get("name"),
        metadata=data.get("metadata"),
        embedding_sizes=data.get("embedding_size"),
    )
    ignored = [name for name in schema_names if name not in columns]
    if ignored:
        # Extra source columns are not implicitly merged into metadata: this
        # would change the explicit contract and could conceal key collisions.
        displayed = [name if len(name) <= 100 else name[:97] + "..." for name in ignored[:10]]
        summary = ", ".join(repr(name) for name in displayed)
        if len(ignored) > len(displayed):
            summary += f", … {len(ignored) - len(displayed)} more"
        dataset.report.warnings.append(
            f"Ignored {len(ignored)} non-contract Parquet column(s): {summary}. "
            "Move fields to metadata to include them in the dataset."
        )
    return dataset


def write_dataset(
    output_dir: str | Path,
    ids: list[str],
    embeddings: Any,
    manifest: DatasetManifest,
    names: list[str | None] | None = None,
    metadata: list[dict[str, Any]] | None = None,
    overwrite: bool = False,
) -> Path:
    """Validate, then write canonical files; existing datasets require opt-in.

    Invalid data writes nothing. Files are staged together and rolled back if
    committing fails. This is not a concurrent-writer transaction or a guarantee
    against process termination between the two final filesystem operations.
    """
    dataset = _validate(ids, embeddings, manifest, names, metadata)
    output_dir = Path(output_dir).expanduser()
    filenames = ("dataset.parquet", "manifest.json")
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"output path is not a directory: {output_dir}")
    for filename in filenames:
        target = output_dir / filename
        if target.exists() or target.is_symlink():
            if not overwrite:
                raise FileExistsError(f"refusing to overwrite {target}; pass overwrite=True explicitly")
            if target.is_dir():
                raise FileExistsError(f"expected a file, found directory: {target}")
    table = pa.table({
        "id": pa.array(dataset.ids, type=pa.string()),
        "name": pa.array(dataset.names, type=pa.string()),
        "embedding": pa.FixedSizeListArray.from_arrays(
            pa.array(dataset.embeddings.reshape(-1), type=pa.float32()),
            manifest.embedding_dim,
        ),
        "metadata": pa.array([
            json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            for value in dataset.metadata
        ], type=pa.string()),
    })
    manifest_text = json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".embedding-dataset-", dir=output_dir.parent) as temporary:
        staged = Path(temporary)
        pq.write_table(table, staged / "dataset.parquet", compression="zstd")
        (staged / "manifest.json").write_text(manifest_text, encoding="utf-8")
        created_dir = not output_dir.exists()
        output_dir.mkdir(exist_ok=True)
        installed: list[str] = []
        backed_up: list[str] = []
        try:
            for filename in filenames:
                target = output_dir / filename
                if overwrite:
                    if target.exists() or target.is_symlink():
                        os.replace(target, staged / (filename + ".backup"))
                        backed_up.append(filename)
                    os.replace(staged / filename, target)
                else:
                    # A hard link fails atomically if a concurrent writer has
                    # created the destination since the earlier existence check.
                    os.link(staged / filename, target)
                installed.append(filename)
        except BaseException:
            for filename in reversed(installed):
                (output_dir / filename).unlink()
            for filename in reversed(backed_up):
                os.replace(staged / (filename + ".backup"), output_dir / filename)
            if created_dir:
                try:
                    output_dir.rmdir()
                except OSError:
                    pass
            raise
    return output_dir
