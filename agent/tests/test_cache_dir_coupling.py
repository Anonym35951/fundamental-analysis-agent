"""Regressionstests für die CACHE_DIR-Kopplung (LAUNCH.md P0-3b).

Ein Render Persistent Disk wird über die Env-Var CACHE_DIR eingebunden.
Historisch hat SecSource diese Variable ignoriert und immer in den
relativen Default "cache/sec" geschrieben — damit wäre der SEC-Cache
trotz Disk ephemer geblieben. Diese Tests erzwingen, dass beide
Cache-Bäume demselben konfigurierten Wurzelverzeichnis folgen.
"""

import os

from agent.DataLoader import DataLoader


def test_sec_source_cache_dir_follows_cache_dir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")

    loader = DataLoader()

    assert loader.cache_dir == str(tmp_path)
    assert loader.sec_source.cache_dir == os.path.join(str(tmp_path), "sec")
    assert os.path.isdir(loader.sec_source.cache_dir)


def test_sec_source_cache_dir_default_unchanged(tmp_path, monkeypatch):
    # Ohne CACHE_DIR muss alles beim alten relativen Default bleiben
    # (Dev-Umgebung, SEC_Tests-Skripte) — hier via cwd-Wechsel isoliert.
    monkeypatch.delenv("CACHE_DIR", raising=False)
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)

    loader = DataLoader()

    assert loader.cache_dir == "cache"
    assert loader.sec_source.cache_dir == os.path.join("cache", "sec")
