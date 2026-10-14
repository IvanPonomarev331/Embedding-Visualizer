"""Public dataset contract and validation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class DatasetValidationError(ValueError):
    """An invalid manifest, table, or row with bounded, actionable diagnostics."""

    def __init__(self, issues: str | list[str], total_issues: int | None = None):
        self.issues = [issues] if isinstance(issues, str) else list(issues)
        self.total_issues = len(self.issues) if total_issues is None else total_issues
        message = "Dataset validation failed:\n" + "\n".join(
            f"- {issue}" for issue in self.issues
        )
        if self.total_issues > len(self.issues):
            message += f"\n- … {self.total_issues - len(self.issues)} more issues"
        super().__init__(message)


@dataclass(frozen=True)
class DatasetManifest:
    """Identity and geometry of a single compatible embedding space."""

    dataset_name: str
    space_id: str
    version: str
    embedding_dim: int
    default_metric: str = "cosine"
    schema_version: int = 1

    def __post_init__(self) -> None:
        issues: list[str] = []
        for name in ("dataset_name", "space_id", "version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                issues.append(f"manifest.{name} must be a nonempty string")
        if (
            not isinstance(self.embedding_dim, int)
            or isinstance(self.embedding_dim, bool)
            or self.embedding_dim <= 0
        ):
            issues.append("manifest.embedding_dim must be a positive integer")
        if self.default_metric not in ("cosine", "euclidean", "dot_product"):
            issues.append(
                "manifest.default_metric must be cosine, euclidean, or dot_product"
            )
        if type(self.schema_version) is not int or self.schema_version != 1:
            issues.append("manifest.schema_version must be integer 1")
        if issues:
            raise DatasetValidationError(issues)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DatasetManifest:
        if not isinstance(value, dict):
            raise DatasetValidationError("manifest must be a JSON object")
        required = {"dataset_name", "space_id", "version", "embedding_dim"}
        allowed = required | {"default_metric", "schema_version"}
        missing = required - value.keys()
        extra = value.keys() - allowed
        issues: list[str] = []
        if missing:
            issues.append("manifest missing fields: " + ", ".join(sorted(missing)))
        if extra:
            issues.append("manifest has unknown fields: " + ", ".join(sorted(map(str, extra))))
        if issues:
            raise DatasetValidationError(issues)
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "space_id": self.space_id,
            "version": self.version,
            "embedding_dim": self.embedding_dim,
            "default_metric": self.default_metric,
            "schema_version": self.schema_version,
        }


@dataclass
class ValidationReport:
    row_count: int
    embedding_dim: int
    zero_vector_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Top-level metadata field -> sorted observed JSON types. Missing keys are
    # absent, whereas an explicit JSON null is reported as "null".
    metadata_fields: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Dataset:
    ids: list[str]
    names: list[str | None]
    embeddings: np.ndarray
    metadata: list[dict[str, Any]]
    manifest: DatasetManifest
    report: ValidationReport
