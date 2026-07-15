"""Regressionstests für EVOLVING.md EV-011: der wöchentliche Symbol-Sync-
Worker (api/services/symbol_sync_service.py::symbol_sync_worker) muss den
Import UNBEDINGT auslösen - anders als der Startup-Check (EV-010) prüft er
KEINEN Schwellwert. Würde er das tun, würde nach dem ersten erfolgreichen
Import (Tabelle dauerhaft > SYMBOL_IMPORT_MIN_COUNT) nie wieder ein Sync
laufen und neue Listings/Delistings blieben für immer unerkannt.
"""
import asyncio

import pytest

from api.crud.favorite import get_favorites
from api.models.base import Base
from api.models.favorite import Favorite
from api.models.symbol import Symbol
from api.models.user import User
import api.services.symbol_sync_service as sync_service
from scripts.import_symbols import upsert_symbols


@pytest.fixture
def symbols_db(db):
    Base.metadata.create_all(db.get_bind(), tables=[Symbol.__table__])
    return db


def _seed_active_symbols(db, count):
    db.add_all(
        [Symbol(symbol=f"SYM{i}", name=f"Company {i}", exchange="NASDAQ") for i in range(count)]
    )
    db.commit()


def test_worker_runs_import_even_when_table_is_far_above_threshold(symbols_db, monkeypatch):
    _seed_active_symbols(symbols_db, 5000)  # weit oberhalb SYMBOL_IMPORT_MIN_COUNT
    calls = []

    async def _record_run(session_factory):
        calls.append(1)
        return {"fetched": 0}

    monkeypatch.setattr(sync_service, "_run_import_locked", _record_run)

    async def scenario():
        task = asyncio.create_task(
            sync_service.symbol_sync_worker(lambda: symbols_db, interval_seconds=0.01)
        )
        await asyncio.sleep(0.05)  # genug Zeit fuer mehrere Ticks
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert len(calls) >= 2  # laeuft wiederholt, nicht nur einmalig


def test_worker_waits_full_interval_before_first_run(symbols_db, monkeypatch):
    calls = []

    async def _record_run(session_factory):
        calls.append(1)

    monkeypatch.setattr(sync_service, "_run_import_locked", _record_run)

    async def scenario():
        task = asyncio.create_task(
            sync_service.symbol_sync_worker(lambda: symbols_db, interval_seconds=10)
        )
        await asyncio.sleep(0.05)  # deutlich kuerzer als das 10s-Intervall
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert calls == []


def test_favorite_survives_symbol_being_delisted_by_sync(symbols_db):
    """Zu schuetzende bestehende Funktion (EV-011-Plan): der woechentliche
    Sync markiert nicht mehr gelistete Symbole via upsert_symbols als
    is_active=False - das darf gespeicherte Favoriten nicht verschwinden
    lassen, da Favorite/get_favorites is_active gar nicht referenzieren
    (unabhaengige Tabellen, nur ueber den Symbol-String lose verknuepft)."""
    Base.metadata.create_all(symbols_db.get_bind(), tables=[Favorite.__table__])
    symbols_db.add(Symbol(symbol="ACME", name="Acme Corp", exchange="NASDAQ", is_active=True))
    user = User(email="fav-user@example.com", hashed_password="x")
    symbols_db.add(user)
    symbols_db.commit()
    symbols_db.refresh(user)
    symbols_db.add(Favorite(user_id=user.id, symbol="ACME"))
    symbols_db.commit()

    upsert_symbols(symbols_db, rows=[])  # ACME taucht in keiner Boersen-Liste mehr auf -> delisted

    acme = symbols_db.query(Symbol).filter(Symbol.symbol == "ACME").one()
    assert acme.is_active is False

    favorites = get_favorites(symbols_db, user.id)
    assert [f.symbol for f in favorites] == ["ACME"]
