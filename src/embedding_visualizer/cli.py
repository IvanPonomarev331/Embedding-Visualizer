"""Local command-line entry points. No framework imports are needed for --help."""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

from embedding_visualizer.settings import load_settings, override_server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect embedding datasets locally.")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Start the local dataset inspector")
    serve.add_argument("--config", type=Path)
    serve.add_argument("--data-dir", type=Path)
    serve.add_argument("--cache-dir", type=Path)
    serve.add_argument("--address")
    serve.add_argument("--port", type=int)

    validate = sub.add_parser("validate", help="Validate a dataset directory or Parquet file")
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--json", action="store_true", help="Print the validation summary as JSON")

    convert = sub.add_parser("convert-npy", help="Convert vectors and metadata to the dataset contract")
    convert.add_argument("--embeddings", type=Path, required=True)
    convert.add_argument("--metadata", type=Path, required=True)
    convert.add_argument("--ids", type=Path, help="JSON or NumPy vector-row IDs; metadata is aligned by ID")
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--dataset-name", required=True)
    convert.add_argument("--space-id", required=True)
    convert.add_argument("--version", required=True)
    convert.add_argument("--metric", choices=["cosine", "euclidean", "dot_product"], default="cosine")
    convert.add_argument("--id-column", default="id")
    convert.add_argument("--name-column", default="name")
    convert.add_argument("--assume-row-order", action="store_true", help="Confirm metadata rows match vector rows when --ids is absent")
    convert.add_argument("--overwrite", action="store_true")

    yt = sub.add_parser("import-yt", help="Import a bounded static YT table snapshot")
    yt.add_argument("--proxy", required=True)
    yt.add_argument("--table", required=True)
    yt.add_argument("--output", type=Path, required=True)
    yt.add_argument("--dataset-name", required=True)
    yt.add_argument("--space-id", required=True)
    yt.add_argument("--version", required=True)
    yt.add_argument("--metric", choices=["cosine", "euclidean", "dot_product"], default="cosine")
    yt.add_argument("--id-column", default="id")
    yt.add_argument("--embedding-column", default="embedding")
    yt.add_argument("--name-column")
    yt.add_argument("--metadata-column")
    yt.add_argument("--embedding-size-column")
    yt.add_argument("--limit", type=int, default=10000)
    yt.add_argument("--offset", type=int, default=0)
    yt.add_argument("--overwrite", action="store_true")

    demo = sub.add_parser("demo", help="Create a small synthetic dataset")
    demo.add_argument("--output", type=Path, default=Path("data/example"))
    demo.add_argument("--overwrite", action="store_true")
    return parser


def _serve(args) -> int:
    config = args.config
    if config is None and Path("visualizer.toml").is_file():
        config = Path("visualizer.toml")
    settings = override_server(load_settings(config), args.address, args.port)
    for key in ("data_dir", "cache_dir"):
        if (value := getattr(args, key)) is not None:
            settings = replace(settings, **{key: value.expanduser().resolve()})
    command = [sys.executable, "-m", "embedding_visualizer.web.server",
               "--address", settings.address, "--port", str(settings.port),
               "--data-dir", str(settings.data_dir), "--cache-dir", str(settings.cache_dir)]
    environment = dict(os.environ)
    environment["NUMBA_CACHE_DIR"] = str(settings.cache_dir / "numba")
    return subprocess.call(command, env=environment)


def _import_yt(args) -> int:
    from embedding_visualizer.adapters.yt import import_yt_dataset

    columns = {name: getattr(args, f"{name}_column") for name in
               ("id", "embedding", "name", "metadata", "embedding_size")
               if getattr(args, f"{name}_column") is not None}
    path = import_yt_dataset(args.proxy, args.table, args.output, columns=columns,
        dataset_name=args.dataset_name, space_id=args.space_id, version=args.version,
        default_metric=args.metric, limit=args.limit, offset=args.offset, overwrite=args.overwrite)
    print(f"Created local YT snapshot: {path}")
    return 0


def _validate(args) -> int:
    from embedding_visualizer.data import load_dataset

    dataset = load_dataset(args.dataset.expanduser())
    summary = {"manifest": dataset.manifest.to_dict(), "validation": asdict(dataset.report)}
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Valid: {dataset.manifest.dataset_name}")
        print(f"Rows: {len(dataset.ids)}; dimensions: {dataset.manifest.embedding_dim}")
        print(f"Space: {dataset.manifest.space_id}; version: {dataset.manifest.version}")
        print(f"Metadata fields: {', '.join(dataset.report.metadata_fields) or 'none'}")
        for warning in dataset.report.warnings:
            print(f"Warning: {warning}")
    return 0


def _convert(args) -> int:
    import numpy as np
    from embedding_visualizer.adapters.npy import convert_npy
    from embedding_visualizer.data import DatasetManifest

    path = args.embeddings.expanduser()
    try:
        matrix = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"Cannot read embedding matrix {path}: {exc}") from exc
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Embedding file must be a two-dimensional .npy array")
    manifest = DatasetManifest(dataset_name=args.dataset_name, space_id=args.space_id,
                               version=args.version, embedding_dim=int(matrix.shape[1]),
                               default_metric=args.metric)
    del matrix
    result = convert_npy(path, args.metadata.expanduser(), args.output.expanduser(), manifest,
                         ids_path=args.ids.expanduser() if args.ids else None,
                         id_column=args.id_column, name_column=args.name_column,
                         assume_row_order=args.assume_row_order, overwrite=args.overwrite)
    print(f"Created dataset: {result}")
    return 0


def _demo(args) -> int:
    import numpy as np
    from embedding_visualizer.data import DatasetManifest, write_dataset

    matrix = np.array([[1, 0, .1, 0], [.9, .1, 0, 0], [0, 1, 0, .1],
                       [.1, .9, 0, 0], [0, 0, 1, .1], [0, .1, .9, 0]], dtype=np.float32)
    manifest = DatasetManifest("Example embeddings", "synthetic-demo-v1", "1", 4)
    result = write_dataset(args.output.expanduser(), ids=[f"example-{i:03}" for i in range(6)],
                           embeddings=matrix, manifest=manifest,
                           names=[f"Example object {i + 1}" for i in range(6)],
                           metadata=[{"group": "ABC"[i // 2], "frequency": (i + 1) * 10,
                                      "active": i % 2 == 0, "tags": ["demo", "synthetic"]}
                                     for i in range(6)], overwrite=args.overwrite)
    print(f"Created synthetic dataset: {result}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return {"serve": _serve, "validate": _validate, "convert-npy": _convert,
                "import-yt": _import_yt, "demo": _demo}[args.command](args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
