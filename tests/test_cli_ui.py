"""Public entry-point checks for real datasets and the local HTTP launch."""

import json
from pathlib import Path
import sys

import pytest

from embedding_visualizer import cli


@pytest.fixture
def example_dataset(tmp_path, capsys):
    output = tmp_path / "example dataset"
    assert cli.main(["demo", "--output", str(output)]) == 0
    capsys.readouterr()
    return output


def test_demo_is_a_valid_portable_dataset(example_dataset, capsys):
    assert (example_dataset / "dataset.parquet").is_file()
    assert (example_dataset / "manifest.json").is_file()
    assert cli.main(["validate", str(example_dataset), "--json"]) == 0
    captured = capsys.readouterr()
    assert not captured.err
    result = json.loads(captured.out)
    assert result["manifest"]["embedding_dim"] == 4
    assert result["validation"]["row_count"] == 6
    assert result["validation"]["zero_vector_ids"] == []
    assert result["validation"]["metadata_fields"] == {
        "active": ["boolean"],
        "frequency": ["integer"],
        "group": ["string"],
        "tags": ["array"],
    }


@pytest.mark.parametrize(
    "contents, diagnostic",
    [
        ("{broken json", "cannot read manifest"),
        ('{"dataset_name":"broken"}', "missing fields"),
        ("[]", "JSON object"),
    ],
)
def test_cli_reports_external_invalid_manifest_without_traceback(
    example_dataset, capsys, contents, diagnostic
):
    (example_dataset / "manifest.json").write_text(contents, encoding="utf-8")
    assert cli.main(["validate", str(example_dataset), "--json"]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "Error:" in captured.err
    assert diagnostic in captured.err
    assert "Traceback" not in captured.err


def test_serve_forwards_config_paths_and_cli_overrides(tmp_path, monkeypatch):
    config_dir = tmp_path / "configuration folder"
    config_dir.mkdir()
    config = config_dir / "visualizer.toml"
    config.write_text(
        'data_dir = "input data"\ncache_dir = "cached results"\nport = 8510\n',
        encoding="utf-8",
    )
    for setting in ("DATA_DIR", "CACHE_DIR", "ADDRESS", "PORT"):
        monkeypatch.delenv(f"VISUALIZER_{setting}", raising=False)
    calls = []

    def capture_launch(command, *, env):
        calls.append(command)
        assert env["NUMBA_CACHE_DIR"] == str(override_cache / "numba")
        return 17

    monkeypatch.setattr(cli.subprocess, "call", capture_launch)
    override_cache = tmp_path / "override cache"
    result = cli.main([
        "serve", "--config", str(config), "--port", "8520",
        "--cache-dir", str(override_cache),
    ])
    assert result == 17
    assert len(calls) == 1
    command = calls[0]
    assert command[:3] == [sys.executable, "-m", "embedding_visualizer.web.server"]
    assert command[command.index("--address") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8520"
    app_options = command
    assert app_options[app_options.index("--data-dir") + 1] == str(config_dir / "input data")
    assert app_options[app_options.index("--cache-dir") + 1] == str(override_cache)


def test_yt_cli_forwards_explicit_columns(tmp_path, monkeypatch, capsys):
    from embedding_visualizer.adapters import yt
    calls = []
    monkeypatch.setattr(yt, "import_yt_dataset", lambda *args, **kwargs: calls.append((args, kwargs)) or tmp_path)
    assert cli.main(["import-yt", "--proxy", "cluster", "--table", "//table", "--output", str(tmp_path),
                     "--dataset-name", "YT", "--space-id", "s", "--version", "v",
                     "--id-column", "hid", "--embedding-column", "vec", "--limit", "25", "--offset", "5"]) == 0
    assert calls[0][1]["columns"] == {"id": "hid", "embedding": "vec"}
    assert calls[0][1]["limit"] == 25
    assert calls[0][1]["offset"] == 5
    assert "Created local YT snapshot" in capsys.readouterr().out
