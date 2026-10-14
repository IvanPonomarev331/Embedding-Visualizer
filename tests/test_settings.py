from pathlib import Path

import pytest

from embedding_visualizer.settings import load_settings


def test_config_relative_paths_and_environment_override(tmp_path, monkeypatch):
    config = tmp_path / "visualizer.toml"
    config.write_text('data_dir = "datasets"\ncache_dir = "cache"\nport = 8510\n')
    monkeypatch.setenv("VISUALIZER_CACHE_DIR", str(tmp_path / "external-cache"))
    settings = load_settings(config)
    assert settings.data_dir == tmp_path / "datasets"
    assert settings.cache_dir == tmp_path / "external-cache"
    assert settings.port == 8510
    assert not settings.data_dir.exists()
    assert not settings.cache_dir.exists()


@pytest.mark.parametrize("content", ['port = 0', 'port = true', 'data_dir = 123', 'unknown = 1'])
def test_invalid_config_is_explicit(tmp_path, content):
    config = tmp_path / "visualizer.toml"
    config.write_text(content)
    with pytest.raises(ValueError):
        load_settings(config)
