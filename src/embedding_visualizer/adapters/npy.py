"""Import a NumPy embedding matrix and explicitly aligned metadata."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from embedding_visualizer.data import (
    DatasetManifest,
    DatasetValidationError,
    write_dataset,
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        if path.suffix.lower() == ".parquet":
            table = pq.read_table(path)
            if len(set(table.column_names)) != len(table.column_names):
                raise DatasetValidationError("Parquet metadata must have unique column names.")
            rows = table.to_pylist()
        elif path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                fields = reader.fieldnames
                if not fields or len(set(fields)) != len(fields):
                    raise DatasetValidationError(
                        "CSV metadata needs a header with unique column names."
                    )
                rows = list(reader)
                if any(None in row for row in rows):
                    raise DatasetValidationError(
                        "A CSV metadata row contains more values than the header."
                    )
        elif path.suffix.lower() == ".json":
            rows = json.loads(path.read_text(encoding="utf-8"))
        else:
            raise DatasetValidationError(
                "Metadata must be a .parquet, .csv, or .json file."
            )
    except DatasetValidationError:
        raise
    except (OSError, ValueError, UnicodeError, csv.Error, pa.ArrowException) as exc:
        raise DatasetValidationError(f"Cannot read metadata from {path}: {exc}") from exc

    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise DatasetValidationError("Metadata must contain a list of row objects.")
    return rows


def _normalize_ids(values: list[Any], source: str) -> list[str]:
    """Normalize only integer IDs; preserve string IDs exactly, including zeroes."""
    result: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
            value = str(value)
        if not isinstance(value, str) or not value.strip():
            raise DatasetValidationError(
                f"{source} ID at row {index} must be a non-empty string or integer; "
                "boolean and floating-point IDs are not supported."
            )
        if value in seen:
            raise DatasetValidationError(
                f"Duplicate {source} ID after integer-to-string conversion: {value!r}."
            )
        seen.add(value)
        result.append(value)
    return result


def _read_ids(path: Path) -> list[str]:
    try:
        if path.suffix.lower() == ".json":
            values = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".npy":
            array = np.load(path, allow_pickle=False)
            if not isinstance(array, np.ndarray) or array.ndim != 1:
                raise DatasetValidationError("The ID .npy file must be a 1D array.")
            values = array.tolist()
        else:
            raise DatasetValidationError("The ID file must be a .json list or .npy array.")
    except DatasetValidationError:
        raise
    except (OSError, ValueError, UnicodeError, EOFError) as exc:
        raise DatasetValidationError(f"Cannot read IDs from {path}: {exc}") from exc
    if not isinstance(values, list):
        raise DatasetValidationError("The ID JSON file must contain a list of IDs.")
    return _normalize_ids(values, "vector")


def _row_metadata(row: dict[str, Any], id_column: str, name_column: str) -> dict[str, Any]:
    nested = row.get("metadata")
    if nested is None or nested == "":
        nested = {}
    elif isinstance(nested, str):
        try:
            nested = json.loads(nested)
        except (ValueError, TypeError) as exc:
            raise DatasetValidationError(
                f"Invalid metadata JSON for ID {row[id_column]!r}: {exc}"
            ) from exc
    if not isinstance(nested, Mapping):
        raise DatasetValidationError(
            f"The metadata field for ID {row[id_column]!r} must be an object."
        )
    result = dict(nested)
    for key, value in row.items():
        if key in {id_column, name_column, "metadata"}:
            continue
        if key in result:
            raise DatasetValidationError(
                f"Metadata key {key!r} for ID {row[id_column]!r} exists in both "
                "the nested metadata object and a separate column."
            )
        result[key] = value
    return result


def convert_npy(
    embeddings_path: Path,
    metadata_path: Path,
    output_dir: Path,
    manifest: DatasetManifest,
    ids_path: Path | None = None,
    id_column: str = "id",
    name_column: str = "name",
    assume_row_order: bool = False,
    overwrite: bool = False,
) -> Path:
    """Convert an embedding matrix plus metadata into a canonical dataset directory.

    An explicit ID file identifies every matrix row. Metadata is reordered to match
    those IDs, and the two ID sets must match exactly. Without an ID file, callers
    must acknowledge that metadata row order matches matrix row order.

    Integer IDs are converted losslessly to strings at this adapter boundary.
    String IDs, including leading zeroes, are preserved. NumPy files never enable
    pickle. CSV extra fields remain strings; use Parquet/JSON for typed metadata.
    """
    if ids_path is None and not assume_row_order:
        raise DatasetValidationError(
            "Matrix row identity is unknown: provide an ID file or explicitly "
            "set assume_row_order=True after checking metadata row order."
        )
    if not id_column or not name_column or id_column == name_column:
        raise DatasetValidationError("ID and name column names must be non-empty and different.")
    if "metadata" in {id_column, name_column}:
        raise DatasetValidationError("The column name 'metadata' is reserved for metadata objects.")

    embeddings_path = Path(embeddings_path).expanduser()
    if embeddings_path.suffix.lower() != ".npy":
        raise DatasetValidationError("The embedding matrix must be a .npy file.")
    try:
        embeddings = np.load(embeddings_path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError, EOFError) as exc:
        raise DatasetValidationError(
            f"Cannot read the embedding matrix from {embeddings_path}: {exc}"
        ) from exc
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise DatasetValidationError("The embedding matrix must be a 2D NumPy array.")

    rows = _read_rows(Path(metadata_path).expanduser())
    if any(id_column not in row for row in rows):
        raise DatasetValidationError(f"Every metadata row must contain ID column {id_column!r}.")
    metadata_ids = _normalize_ids([row[id_column] for row in rows], "metadata")
    if ids_path is not None:
        ids = _read_ids(Path(ids_path).expanduser())
        if len(ids) != embeddings.shape[0]:
            raise DatasetValidationError(
                f"ID count ({len(ids)}) differs from matrix row count ({embeddings.shape[0]})."
            )
        missing = set(ids) - set(metadata_ids)
        extra = set(metadata_ids) - set(ids)
        if missing or extra:
            raise DatasetValidationError(
                "Vector and metadata IDs must match exactly: "
                f"{len(missing)} missing metadata IDs, {len(extra)} extra metadata IDs."
            )
        by_id = dict(zip(metadata_ids, rows, strict=True))
        rows = [by_id[value] for value in ids]
    else:
        ids = metadata_ids
        if len(rows) != embeddings.shape[0]:
            raise DatasetValidationError(
                f"Metadata row count ({len(rows)}) differs from matrix row count "
                f"({embeddings.shape[0]})."
            )

    names = [row.get(name_column) for row in rows]
    metadata = [_row_metadata(row, id_column, name_column) for row in rows]
    return write_dataset(
        Path(output_dir),
        ids,
        embeddings,
        manifest,
        names=names,
        metadata=metadata,
        overwrite=overwrite,
    )
