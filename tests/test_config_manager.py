"""Tests for ConfigurationManager config file discovery."""

from pathlib import Path

from on1builder.config.manager import ConfigurationManager


def test_find_config_file_prefers_home(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    home_env = tmp_path / ".env"
    home_env.write_text("TEST=1\n", encoding="utf-8")

    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: project))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    manager = ConfigurationManager()
    result = manager._find_config_file()
    assert result == home_env
