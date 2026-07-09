import os
import shutil
from pathlib import Path

import pytest

from agent.Model import Model

FIXTURES_CACHE_DIR = Path(__file__).parent / "fixtures" / "cache"


@pytest.fixture
def frozen_cache_dir(tmp_path):
    """Kopiert die eingefrorenen Fixture-Dateien (echte, früher abgerufene
    SEC-Daten aus dem Produktions-Cache) in ein temporäres Verzeichnis und
    setzt ihre mtime auf "jetzt".

    Grund: DataLoader._load_cached_data prüft Frische anhand der Datei-mtime,
    nicht anhand von Daten im Inhalt (siehe agent/DataLoader.py). Ohne dieses
    Auffrischen würden manche Cache-Keys (z. B. "balance_sheet_annual", TTL
    10s) als abgelaufen gelten und einen echten Netzwerk-Call auslösen — dann
    wären die Tests weder deterministisch noch netzwerkfrei.
    """
    for fixture_file in FIXTURES_CACHE_DIR.glob("*.json"):
        target = tmp_path / fixture_file.name
        shutil.copy(fixture_file, target)
        os.utime(target, None)
    return tmp_path


@pytest.fixture
def model(frozen_cache_dir):
    """Model-Instanz, deren DataLoader ausschließlich aus den eingefrorenen
    Fixtures liest — kein Netzwerkzugriff in Tests."""
    m = Model()
    m.dataloader.cache_dir = str(frozen_cache_dir)
    return m
