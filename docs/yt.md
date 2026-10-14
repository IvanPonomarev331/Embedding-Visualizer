# Import from YT

The optional adapter downloads a bounded slice of a **static** YT table and saves
the same local `dataset.parquet` + `manifest.json` format as the NumPy adapter.
After import, exploration uses the local snapshot and does not access YT.

## Setup

Install the optional SDK in the application's Python environment:

```sh
python -m pip install ytsaurus-client
```

An existing compatible internal `yt.wrapper` installation is also supported.
Local datasets do not require the SDK. Authentication uses the SDK's existing
environment/configuration and normal token discovery. The application does not
provide a token flag, save credentials, or include remote exception details in
its errors. The adapter copies the SDK's environment-initialized configuration
into an independent `YtClient` and applies the explicit proxy.

## Python API

```python
from embedding_visualizer.adapters.yt import import_yt_dataset

import_yt_dataset(
    proxy="your-cluster",
    table_path="//home/your-user/embedding_snapshot",
    output="./data/my-snapshot",
    columns={
        "id": "object_id",
        "name": "title",
        "embedding": "vector",
        "metadata": "attributes",
        "embedding_size": "dimension",
    },
    dataset_name="Product embeddings",
    space_id="encoder-v3/output",
    version="snapshot-2026-10-13",
    default_metric="cosine",
    limit=10_000,
    offset=0,
)
```

Only `id` and `embedding` are required in `columns`; omit any optional mapping
whose source column does not exist. Only mapped source columns are requested.
The source path must be a plain `//path/to/table`; column/range selectors belong
in explicit arguments. A configured `client=` can be injected by SDK users and
tests; that client must target the cluster supplied as `proxy`.

IDs accept non-empty strings or integers. Integer IDs are converted losslessly
to strings, and collisions such as `1` and `"1"` are rejected. String IDs retain
leading zeroes. Embeddings must be numeric vectors with the same nonzero
dimension. `embedding_size`, when mapped, must be an integer matching that
dimension on every row. Metadata accepts an object, a JSON-encoded object, or
null; metadata values must be finite JSON data. Names accept strings or null.
The local dataset validator checks duplicates, dimensions, NaN/Inf, metadata,
and float32 range before writing the snapshot.

## Sampling and limits

`offset` is a zero-based row index; `limit` is a positive upper bound on the
number of rows. The server receives the half-open interval
`[offset, offset + limit)` and explicit column selection through `TablePath`.
The adapter does not read the entire table before applying a local limit, run a
query, or submit a distributed operation. It closes the read stream when done.

This is a deterministic contiguous slice for an unchanged table, **not random
sampling**. A slice near the end can contain fewer than `limit` rows; an empty
slice is an error. A changed or reordered source table can produce different
results for the same offset. Use a fixed upstream snapshot/version when exact
reproducibility is needed. The selected vectors and metadata must fit local RAM.
Dynamic tables require a separate static snapshot prepared outside the app.

## Verification status

The adapter is covered by offline tests with a mocked SDK client: server bounds,
column mapping, static-table checks, validation, stream cleanup, optional SDK
loading, and authentication-detail suppression. No actual YT table was read
during development because no cluster/table was provided; connectivity and
permissions need verification on the first real import.

API behavior was checked against the primary YTsaurus documentation and SDK
source:

- [Python client installation](https://ytsaurus.tech/docs/en/api/python/start)
- [Python API: TablePath, read_table, and independent clients](https://ytsaurus.tech/docs/en/api/python/userdoc)
- [SDK configuration initialization](https://github.com/ytsaurus/ytsaurus/blob/main/yt/python/yt/wrapper/config.py)
- [YtClient TablePath API](https://github.com/ytsaurus/ytsaurus/blob/main/yt/python/yt/wrapper/client_impl.py)
- [Dynamic table row-index limitations](https://ytsaurus.tech/docs/en/user-guide/dynamic-tables/mapreduce)
