"""Regressionstests für EVOLVING.md EV-010: der Symbol-Import darf beim
Server-Start nur laufen, wenn die `symbols`-Tabelle unterhalb des
Schwellwerts liegt (Render-Erststart bzw. eine Prod-DB, auf der
scripts/import_symbols.py nie manuell lief - siehe api/main.py:
sync_symbols_on_startup), und nie parallel zu einem bereits laufenden
Import (api/services/symbol_sync_service.py: _import_lock).
"""
import asyncio

import pytest

from api.models.base import Base
from api.models.symbol import Symbol
import api.services.symbol_sync_service as sync_service
from scripts.import_symbols import RawSymbol


@pytest.fixture
def symbols_db(db):
    Base.metadata.create_all(db.get_bind(), tables=[Symbol.__table__])
    return db


async def _fake_to_thread(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def _seed_active_symbols(db, count):
    db.add_all(
        [Symbol(symbol=f"SYM{i}", name=f"Company {i}", exchange="NASDAQ") for i in range(count)]
    )
    db.commit()


def test_count_active_symbols_ignores_delisted(symbols_db):
    symbols_db.add_all(
        [
            Symbol(symbol="AAPL", name="Apple", exchange="NASDAQ", is_active=True),
            Symbol(symbol="DEAD", name="Delisted Co", exchange="NASDAQ", is_active=False),
        ]
    )
    symbols_db.commit()

    assert sync_service.count_active_symbols(symbols_db) == 1


def test_startup_sync_skips_import_when_above_threshold(symbols_db, monkeypatch):
    monkeypatch.setattr(sync_service, "SYMBOL_IMPORT_MIN_COUNT", 2)
    _seed_active_symbols(symbols_db, 2)

    def _forbid(*args, **kwargs):
        raise AssertionError("Import darf bei ausreichend gefüllter Tabelle nicht laufen")

    monkeypatch.setattr(sync_service, "_run_import_locked", _forbid)

    asyncio.run(sync_service.sync_symbols_on_startup(lambda: symbols_db))


def test_startup_sync_triggers_import_when_below_threshold(symbols_db, monkeypatch):
    monkeypatch.setattr(sync_service, "SYMBOL_IMPORT_MIN_COUNT", 5)
    _seed_active_symbols(symbols_db, 1)
    monkeypatch.setattr(asyncio, "to_thread", _fake_to_thread)
    monkeypatch.setattr(
        sync_service, "fetch_nasdaq_listed", lambda: [RawSymbol("ZZZZ", "Brand New Listing", "NASDAQ")]
    )
    monkeypatch.setattr(sync_service, "fetch_other_listed", lambda: [])

    asyncio.run(sync_service.sync_symbols_on_startup(lambda: symbols_db))

    assert symbols_db.query(Symbol).filter(Symbol.symbol == "ZZZZ").count() == 1


def test_startup_sync_below_threshold_does_not_crash_server_start_on_fetch_error(symbols_db, monkeypatch):
    """Downloadfehler (Netzwerk, NASDAQ-Trader down) duerfen den
    Server-Start nicht verhindern - run_symbol_import wirft, der Fehler
    wird geloggt statt propagiert (Implementierungsschritte EV-010)."""
    monkeypatch.setattr(sync_service, "SYMBOL_IMPORT_MIN_COUNT", 5)
    _seed_active_symbols(symbols_db, 1)
    monkeypatch.setattr(asyncio, "to_thread", _fake_to_thread)

    def _boom():
        raise ConnectionError("NASDAQ Trader nicht erreichbar")

    monkeypatch.setattr(sync_service, "fetch_nasdaq_listed", _boom)

    asyncio.run(sync_service.sync_symbols_on_startup(lambda: symbols_db))  # darf nicht werfen


def test_run_import_locked_skips_second_call_while_first_is_in_flight(symbols_db, monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def _blocking_to_thread(fn, *args, **kwargs):
        calls.append(1)
        started.set()
        await release.wait()
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _blocking_to_thread)
    monkeypatch.setattr(sync_service, "fetch_nasdaq_listed", lambda: [])
    monkeypatch.setattr(sync_service, "fetch_other_listed", lambda: [])

    async def scenario():
        task1 = asyncio.create_task(sync_service._run_import_locked(lambda: symbols_db))
        await started.wait()  # task1 haelt den Lock jetzt sicher

        result2 = await sync_service._run_import_locked(lambda: symbols_db)
        assert result2 is None  # zweiter Aufruf wird abgewiesen, nicht verzoegert/verschlangt

        release.set()
        await task1

    asyncio.run(scenario())
    assert calls == [1]
