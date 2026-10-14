"""Typed top-level metadata discovery and declarative slice evaluation.

Missing keys and explicit JSON null are different values throughout. Filters
never evaluate expressions or alter the dataset's row order.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

import numpy as np

from embedding_visualizer.data.models import Dataset

from .common import ComputeError


_SCALAR_TYPES = {"null", "boolean", "integer", "number", "string"}
_KINDS = {"numeric", "categorical", "boolean", "mixed"}


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        if not math.isfinite(value):
            raise ComputeError("Metadata and filter values must contain finite JSON numbers.")
        return "number"
    if type(value) is str:
        return "string"
    if type(value) is list:
        return "array"
    if type(value) is dict:
        return "object"
    raise ComputeError(f"Unsupported metadata value type: {type(value).__name__}.")


def _check_dataset(dataset: Dataset) -> None:
    if not isinstance(dataset, Dataset):
        raise ComputeError("dataset must be a Dataset.")
    count = len(dataset.ids)
    if len(dataset.names) != count or len(dataset.metadata) != count:
        raise ComputeError("Dataset IDs, names and metadata must contain the same rows.")
    if any(not isinstance(identifier, str) for identifier in dataset.ids):
        raise ComputeError("Dataset IDs must be strings.")
    if any(name is not None and not isinstance(name, str) for name in dataset.names):
        raise ComputeError("Dataset names must be strings or null.")
    if any(not isinstance(row, dict) or any(not isinstance(key, str) for key in row) for row in dataset.metadata):
        raise ComputeError("Metadata rows must be objects with string field names.")


def infer_fields(dataset: Dataset, max_values: int = 200) -> list[dict[str, Any]]:
    """Describe observed metadata fields with bounded, typed scalar choices.

    ``count`` counts present keys, including null; ``missing_count`` counts
    absent keys. ``kind`` ignores null: numeric for integers/numbers, boolean
    for booleans, categorical for strings, mixed otherwise. ``min``/``max``
    summarize all numeric values, including in mixed fields. ``values`` are
    scalar frequency records sorted by frequency then type/value; nested
    objects and arrays appear only in ``type_counts``. ``value_count`` and
    ``values_truncated`` disclose when choices have been limited.
    """
    _check_dataset(dataset)
    if isinstance(max_values, (bool, np.bool_)) or not isinstance(max_values, (int, np.integer)) or max_values < 1:
        raise ComputeError("max_values must be a positive integer.")
    observed: dict[str, dict[str, Any]] = {}
    for row in dataset.metadata:
        for name, value in row.items():
            kind = _json_type(value)
            field = observed.setdefault(name, {"types": Counter(), "values": Counter(), "min": None, "max": None})
            field["types"][kind] += 1
            if kind in _SCALAR_TYPES:
                field["values"][(kind, value)] += 1
            if kind in ("integer", "number"):
                field["min"] = value if field["min"] is None else min(field["min"], value)
                field["max"] = value if field["max"] is None else max(field["max"], value)

    result = []
    for name, field in sorted(observed.items()):
        types = field["types"]
        non_null = set(types) - {"null"}
        if non_null and non_null <= {"integer", "number"}:
            kind = "numeric"
        elif non_null == {"boolean"}:
            kind = "boolean"
        elif non_null == {"string"}:
            kind = "categorical"
        else:
            kind = "mixed"
        choices = sorted(field["values"].items(), key=lambda item: (-item[1], item[0][0], repr(item[0][1])))
        count = sum(types.values())
        result.append({
            "name": name,
            "kind": kind,
            "count": count,
            "missing_count": len(dataset.ids) - count,
            "null_count": types.get("null", 0),
            "type_counts": dict(sorted(types.items())),
            "values": [{"type": scalar_type, "value": value, "count": frequency}
                       for (scalar_type, value), frequency in choices[:max_values]],
            "value_count": len(choices),
            "values_truncated": len(choices) > max_values,
            "min": field["min"],
            "max": field["max"],
        })
    return result


def _typed_choice(value: Any) -> tuple[str, Any]:
    if isinstance(value, dict):
        if not {"type", "value"} <= value.keys() or value.keys() - {"type", "value", "count"}:
            raise ComputeError("A typed filter value must contain type/value and may include count.")
        if "count" in value and (type(value["count"]) is not int or value["count"] < 0):
            raise ComputeError("A typed value's count must be a nonnegative integer.")
        actual = _json_type(value["value"])
        # JavaScript serializes an integral Number such as 1.0 as JSON 1.
        # The explicit descriptor preserves its original metadata type.
        if value["type"] == "number" and actual == "integer":
            try:
                restored = float(value["value"])
            except OverflowError as exc:
                raise ComputeError("Filter number is outside the finite range.") from exc
            if not math.isfinite(restored):
                raise ComputeError("Filter number must be finite.")
            return "number", restored
        if not isinstance(value["type"], str) or value["type"] not in _SCALAR_TYPES or value["type"] != actual:
            raise ComputeError("A typed filter value's type must match its scalar JSON value.")
        return actual, value["value"]
    actual = _json_type(value)
    if actual not in _SCALAR_TYPES:
        raise ComputeError("The in operator accepts only JSON scalar values, not arrays or objects.")
    return actual, value


def _numeric_bound(value: Any, name: str) -> int | float:
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ComputeError(f"{name} must be a finite number, not a boolean or string.")
    return value


def apply_filters(dataset: Dataset, filters: list[dict[str, Any]], search: str = "") -> np.ndarray:
    """Return an AND-combined row mask from validated declarative filters.

    Every filter requires ``field`` and ``op``. Optional ``kind`` is a validated
    UI hint (numeric/categorical/boolean/mixed), not a coercion instruction.

    * ``in`` requires ``values``: raw JSON scalars or typed records returned by
      :func:`infer_fields`. Values compare by JSON type, so true, 1, 1.0 and
      "1" remain distinct. An empty list matches no present values.
    * ``range`` requires at least one finite ``min``/``max`` and matches numeric
      values inclusively, excluding booleans, nulls and other types.
    * ``exists`` matches key presence, including explicit null.
    * ``missing`` matches absent keys only.

    ``in`` and ``range`` additionally accept ``missing: true`` to include absent
    keys. Explicit null can be selected with ``in`` and value null. Unknown
    fields, operators and unused parameters raise ComputeError. Search is an
    additional case-insensitive substring condition over ID or name. No filters
    and empty search include every row; neither inputs nor row order change.
    """
    _check_dataset(dataset)
    if not isinstance(filters, list):
        raise ComputeError("filters must be a list of filter objects.")
    if not isinstance(search, str):
        raise ComputeError("search must be a string.")
    fields = {field for row in dataset.metadata for field in row}
    compiled = []
    for position, spec in enumerate(filters):
        prefix = f"Filter {position + 1}"
        if not isinstance(spec, dict):
            raise ComputeError(f"{prefix} must be an object.")
        field = spec.get("field")
        op = spec.get("op")
        if not isinstance(field, str) or field not in fields:
            raise ComputeError(f"{prefix} refers to an unknown metadata field: {field!r}.")
        if not isinstance(op, str) or op not in ("in", "range", "exists", "missing"):
            raise ComputeError(f"{prefix} op must be in, range, exists, or missing.")
        if "kind" in spec and (not isinstance(spec["kind"], str) or spec["kind"] not in _KINDS):
            raise ComputeError(f"{prefix} kind must be numeric, categorical, boolean, or mixed.")
        allowed = {"field", "op", "kind"}
        if op == "in":
            allowed |= {"values", "missing"}
        elif op == "range":
            allowed |= {"min", "max", "missing"}
        extras = spec.keys() - allowed
        if extras:
            raise ComputeError(f"{prefix} contains unsupported parameters: {', '.join(sorted(map(str, extras)))}.")
        include_missing = spec.get("missing", False)
        if type(include_missing) is not bool:
            raise ComputeError(f"{prefix} missing must be a boolean.")
        values = None
        lower = upper = None
        if op == "in":
            if not isinstance(spec.get("values"), list):
                raise ComputeError(f"{prefix} in requires a list of values.")
            values = {_typed_choice(value) for value in spec["values"]}
        elif op == "range":
            if "min" not in spec and "max" not in spec:
                raise ComputeError(f"{prefix} range requires min or max.")
            if "min" in spec:
                lower = _numeric_bound(spec["min"], f"{prefix} min")
            if "max" in spec:
                upper = _numeric_bound(spec["max"], f"{prefix} max")
            if lower is not None and upper is not None and lower > upper:
                raise ComputeError(f"{prefix} min must not exceed max.")
        compiled.append((field, op, include_missing, values, lower, upper))

    needle = search.casefold().strip()
    mask = np.ones(len(dataset.ids), dtype=bool)
    for index, row in enumerate(dataset.metadata):
        if needle and needle not in dataset.ids[index].casefold() and needle not in (dataset.names[index] or "").casefold():
            mask[index] = False
            continue
        for field, op, include_missing, choices, lower, upper in compiled:
            present = field in row
            if op == "exists":
                matches = present
            elif op == "missing":
                matches = not present
            elif not present:
                matches = include_missing
            elif op == "in":
                value_type = _json_type(row[field])
                matches = value_type in _SCALAR_TYPES and (value_type, row[field]) in choices
            else:
                value_type = _json_type(row[field])
                matches = value_type in ("integer", "number") and (lower is None or row[field] >= lower) and (upper is None or row[field] <= upper)
            if not matches:
                mask[index] = False
                break
    return mask
