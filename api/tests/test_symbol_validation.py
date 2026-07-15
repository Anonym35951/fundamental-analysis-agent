"""Regressionstests für EVOLVING.md EV-014: ensure_known_symbol
(api/utils/symbol_validation.py) darf gültige Symbole nie blockieren
(inkl. Punkt-Schreibweise wie "BRK.B"), muss unbekannte/delistete Symbole
aber mit einer verständlichen 422-Meldung ablehnen - und darf in einer noch
leeren symbols-Tabelle (kein Import gelaufen) niemand blockieren."""
import pytest
from fastapi import HTTPException

from api.models.base import Base
from api.models.symbol import Symbol
from api.utils.symbol_validation import ensure_known_symbol


@pytest.fixture
def symbols_db(db):
    Base.metadata.create_all(db.get_bind(), tables=[Symbol.__table__])
    return db


def test_known_active_symbol_passes_and_returns_itself(symbols_db):
    symbols_db.add(Symbol(symbol="AAPL", name="Apple Inc.", exchange="NASDAQ", is_active=True))
    symbols_db.commit()

    assert ensure_known_symbol(symbols_db, "AAPL") == "AAPL"


def test_unknown_symbol_is_rejected_with_422(symbols_db):
    symbols_db.add(Symbol(symbol="AAPL", name="Apple Inc.", exchange="NASDAQ", is_active=True))
    symbols_db.commit()

    with pytest.raises(HTTPException) as exc:
        ensure_known_symbol(symbols_db, "FOOBARX")

    assert exc.value.status_code == 422
    assert "nicht im unterstützten" in exc.value.detail


def test_delisted_symbol_is_rejected_with_distinct_message(symbols_db):
    symbols_db.add(Symbol(symbol="DEADCO", name="Delisted Co", exchange="NASDAQ", is_active=False))
    symbols_db.commit()

    with pytest.raises(HTTPException) as exc:
        ensure_known_symbol(symbols_db, "DEADCO")

    assert exc.value.status_code == 422
    assert "nicht mehr gelistet" in exc.value.detail


def test_dot_style_share_class_symbol_matches_dash_style_db_row_and_returns_canonical_spelling(symbols_db):
    """Marktübliche Schreibweise "BRK.B" muss die per
    scripts/import_symbols.py normalisiert gespeicherte Zeile "BRK-B"
    finden (Root Cause, s. EVOLVING.md EV-014 'Zu schützende Funktionen') -
    UND die kanonische Schreibweise zurückgeben, damit der Rest der Anfrage
    (Job, Agent-Aufruf, Historie) mit "BRK-B" statt "BRK.B" weiterarbeitet."""
    symbols_db.add(Symbol(symbol="BRK-B", name="Berkshire Hathaway Inc", exchange="NYSE", is_active=True))
    symbols_db.commit()

    assert ensure_known_symbol(symbols_db, "BRK.B") == "BRK-B"


def test_empty_symbols_table_does_not_block_anything_and_returns_symbol_unchanged(symbols_db):
    """Kein Import je gelaufen (z. B. frische lokale Dev-DB) -> keine
    Validierung, statt jede Analyse zu blockieren; Symbol kommt unveraendert
    zurueck, da keine kanonische Schreibweise ermittelbar ist."""
    assert ensure_known_symbol(symbols_db, "IRGENDWAS") == "IRGENDWAS"
