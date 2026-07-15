"""Regressionstests für EVOLVING.md EV-012: Admin-Endpoints, die beweisen,
wie viele NYSE/NASDAQ-Symbole tatsächlich aktiv/durchsuchbar sind
(GET /admin/symbols/stats), und ein manueller Import-Trigger ohne
Render-Shell-Zugriff (POST /admin/symbols/refresh).

Stil wie test_quota_active_jobs_order_p2_3.py: direkte Funktionsaufrufe
gegen die In-Memory-SQLite-db-Fixture, kein TestClient.
"""
import asyncio

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from api.core.dependencies import require_admin
from api.models.base import Base
from api.models.symbol import Symbol
from api.models.user import User
import api.routes.admin_symbols as admin_symbols_module
from api.routes.admin_symbols import _compute_symbol_stats, get_symbol_stats, refresh_symbols


@pytest.fixture
def symbols_db(db):
    Base.metadata.create_all(db.get_bind(), tables=[Symbol.__table__])
    return db


def _make_user(db, **kwargs) -> User:
    defaults = {"email": f"user{id(kwargs)}@example.com", "hashed_password": "x"}
    defaults.update(kwargs)
    user = User(**defaults)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _fake_request() -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/admin/symbols/stats",
        "headers": [],
        "client": ("testclient", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "query_string": b"",
    }
    return Request(scope)


def _seed(db, exchange, active_count, inactive_count):
    db.add_all(
        [
            Symbol(symbol=f"{exchange}A{i}", name=f"{exchange} Active {i}", exchange=exchange, is_active=True)
            for i in range(active_count)
        ]
        + [
            Symbol(symbol=f"{exchange}I{i}", name=f"{exchange} Inactive {i}", exchange=exchange, is_active=False)
            for i in range(inactive_count)
        ]
    )
    db.commit()


# ---------------------------------------------------------------------------
# _compute_symbol_stats - reine Funktion
# ---------------------------------------------------------------------------

def test_compute_symbol_stats_counts_by_exchange_and_status(symbols_db):
    _seed(symbols_db, "NASDAQ", active_count=3, inactive_count=1)
    _seed(symbols_db, "NYSE", active_count=2, inactive_count=0)

    stats = _compute_symbol_stats(symbols_db)

    assert stats["total"] == 6
    assert stats["active"] == 5
    assert stats["inactive"] == 1
    assert stats["by_exchange"]["NASDAQ"] == {"active": 3, "inactive": 1}
    assert stats["by_exchange"]["NYSE"] == {"active": 2, "inactive": 0}


def test_compute_symbol_stats_empty_table(symbols_db):
    stats = _compute_symbol_stats(symbols_db)

    assert stats == {
        "total": 0,
        "active": 0,
        "inactive": 0,
        "by_exchange": {},
        "last_updated_at": None,
    }


# ---------------------------------------------------------------------------
# require_admin-Gate auf beiden neuen Routen
# ---------------------------------------------------------------------------

def test_get_symbol_stats_rejects_non_admin(symbols_db):
    non_admin = _make_user(symbols_db, plan="free")

    with pytest.raises(HTTPException) as exc:
        require_admin(current_user=non_admin)
    assert exc.value.status_code == 403

    # Die Route selbst haengt require_admin nur als Depends ein - der 403
    # entsteht bereits in der Dependency, bevor get_symbol_stats/refresh_
    # symbols ueberhaupt aufgerufen werden (FastAPI-Standardverhalten).


def test_get_symbol_stats_allows_admin(symbols_db):
    admin = _make_user(symbols_db, plan="admin")
    _seed(symbols_db, "NASDAQ", active_count=2, inactive_count=0)

    result = get_symbol_stats(request=_fake_request(), db=symbols_db, _=admin)

    assert result["active"] == 2


# ---------------------------------------------------------------------------
# refresh_symbols - Hintergrund-Trigger
# ---------------------------------------------------------------------------

def test_refresh_symbols_queues_background_import_and_responds_immediately(symbols_db, monkeypatch):
    admin = _make_user(symbols_db, plan="admin")
    calls = []

    async def _fake_run_import_locked(session_factory):
        calls.append(1)

    monkeypatch.setattr(admin_symbols_module.symbol_sync_service, "_run_import_locked", _fake_run_import_locked)

    async def scenario():
        result = await refresh_symbols(request=_fake_request(), db=symbols_db, _=admin)
        await asyncio.sleep(0.01)  # gibt dem Fire-and-Forget-Task Zeit zu laufen
        return result

    result = asyncio.run(scenario())

    assert result == {"status": "started"}
    assert calls == [1]


def test_refresh_symbols_reports_already_running_without_queuing_twice(symbols_db, monkeypatch):
    admin = _make_user(symbols_db, plan="admin")

    async def _never_returns(session_factory):
        await asyncio.sleep(10)

    monkeypatch.setattr(admin_symbols_module.symbol_sync_service, "_run_import_locked", _never_returns)
    monkeypatch.setattr(admin_symbols_module.symbol_sync_service._import_lock, "locked", lambda: True)

    result = asyncio.run(refresh_symbols(request=_fake_request(), db=symbols_db, _=admin))

    assert result == {"status": "already_running"}
