"""Regressionstests für EVOLVING.md EV-001: Ist-Verhalten von GET
/analyze/symbols (api/routes/analyze.py::search_symbols) vor den
Universum-Erweiterungen (EV-010ff) festhalten.

Stil wie test_favorites_validation_p2_8.py: direkter Funktionsaufruf gegen
die In-Memory-SQLite-db-Fixture, kein TestClient.
"""
import pytest
from starlette.requests import Request

from api.models.base import Base
from api.models.symbol import Symbol
from api.routes.analyze import POPULAR_SYMBOLS, search_symbols


DEFAULT_LIMIT = 20


def _fake_request() -> Request:
    # Wie in test_quota_active_jobs_order_p2_3.py: der @limiter.limit-
    # Dekorator braucht ein echtes Request-Objekt (liest u.a. Header/Client
    # fuers Rate-Limit-Bucketing), ein blosses None reicht nicht.
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/analyze/symbols",
        "headers": [],
        "client": ("testclient", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "query_string": b"",
    }
    return Request(scope)


def _search(db, query="", limit=DEFAULT_LIMIT):
    return search_symbols(request=_fake_request(), query=query, limit=limit, db=db)


@pytest.fixture
def symbols_db(db):
    Base.metadata.create_all(db.get_bind(), tables=[Symbol.__table__])
    db.add_all(
        [
            Symbol(symbol="AAPL", name="Apple Inc.", exchange="NASDAQ"),
            Symbol(symbol="MSFT", name="Microsoft Corporation", exchange="NASDAQ"),
            Symbol(symbol="ZION", name="Zions Bancorporation", exchange="NASDAQ"),
            Symbol(symbol="KO", name="Coca-Cola Co", exchange="NYSE"),
            Symbol(symbol="BRK.B", name="Berkshire Hathaway Inc", exchange="NYSE"),
            Symbol(symbol="DELISTEDCO", name="Delisted Company", exchange="NASDAQ", is_active=False),
        ]
    )
    db.commit()
    return db


def test_empty_query_returns_only_active_popular_symbols(symbols_db):
    results = _search(symbols_db, query="")

    returned = {r["symbol"] for r in results}
    assert returned <= set(POPULAR_SYMBOLS)
    assert "AAPL" in returned  # Teil von POPULAR_SYMBOLS und in der Test-DB aktiv
    assert "DELISTEDCO" not in returned


def test_empty_query_falls_back_to_static_list_when_table_empty(db):
    Base.metadata.create_all(db.get_bind(), tables=[Symbol.__table__])

    results = _search(db, query="", limit=5)

    assert [r["symbol"] for r in results] == POPULAR_SYMBOLS[:5]


def test_query_finds_known_large_cap_by_symbol(symbols_db):
    results = _search(symbols_db, query="AAPL")

    assert any(r["symbol"] == "AAPL" for r in results)


def test_query_finds_small_cap_by_symbol(symbols_db):
    results = _search(symbols_db, query="ZION")

    assert any(r["symbol"] == "ZION" for r in results)


def test_query_finds_symbol_with_special_character(symbols_db):
    results = _search(symbols_db, query="BRK.B")

    assert any(r["symbol"] == "BRK.B" for r in results)


def test_query_matches_company_name(symbols_db):
    results = _search(symbols_db, query="Coca-Cola")

    assert any(r["symbol"] == "KO" for r in results)


def test_query_excludes_inactive_delisted_symbol(symbols_db):
    results = _search(symbols_db, query="DELISTEDCO")

    assert results == []


def test_query_for_unknown_symbol_returns_empty(symbols_db):
    results = _search(symbols_db, query="FOOBARX")

    assert results == []


def test_limit_is_respected(symbols_db):
    results = _search(symbols_db, query="", limit=2)

    assert len(results) <= 2


def test_result_shape_has_symbol_name_sectors(symbols_db):
    results = _search(symbols_db, query="AAPL")

    assert results
    entry = results[0]
    assert set(["symbol", "name", "sectors"]) <= set(entry.keys())
    assert isinstance(entry["sectors"], list)
